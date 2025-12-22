import torch
import math
import numpy as np
import torch.nn as nn
from einops import rearrange
from transformers.cache_utils import DynamicCache
from src.builder import BUILDER
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),)


# 순서 배열에 따라 앞에서부터 mask_len 개수만큼 마스킹
def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len, device=order.device)
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()],
                            src=torch.ones(bsz, seq_len, device=order.device)).bool()
    return masking


class Harmon(nn.Module):
    def __init__(self,
                 vae,
                 vae_scale,
                 llm,
                 mar,
                 tokenizer,
                 prompt_template):
        super().__init__()
        # VAE
        # NOTE : MAR encoder decoder
        self.vae = BUILDER.build(vae)
        self.vae.requires_grad_(False)
        self.vae_scale = vae_scale

        # LLM
        self.llm = BUILDER.build(llm)
        self.tokenizer = BUILDER.build(tokenizer)
        self.prompt_template = prompt_template

        # MAR
        self.mar = BUILDER.build(mar)
        # projection layers
        # self.proj_in : MAR to LLM
        # self.proj_out: LLM to MAR
        self.proj_in = build_mlp(hidden_size=self.mar.encoder_embed_dim,
                                 projector_dim=self.llm.config.hidden_size,
                                 z_dim=self.llm.config.hidden_size)
        self.proj_out = build_mlp(hidden_size=self.llm.config.hidden_size,
                                  projector_dim=self.llm.config.hidden_size,
                                  z_dim=self.mar.encoder_embed_dim)

    @property
    def llm_model(self):
        return self.llm.model

    @property
    def device(self):
        return self.llm.device

    @property
    def dtype(self):
        return self.llm.dtype

    @property
    def gen_seq_len(self):
        return self.mar.seq_len

    @property
    def token_embed_dim(self):
        return self.vae.embed_dim * (self.mar.patch_size ** 2)

    '''
    x : [B, C, H, W] (pixel space)
    z : [B, M, N, D] (latent space) & D = c * p * q
    '''
    @torch.no_grad()
    def encode(self, x):
        posterior = self.vae.encode(x)
        z = posterior.mode().mul_(self.vae_scale)
        z = rearrange(z, 'b c (m p) (n q) -> b m n (c p q)',
                      p=self.mar.patch_size, q=self.mar.patch_size)

        return z

    @torch.no_grad()
    def decode(self, z):
        z /= self.vae_scale
        z = rearrange(z, 'b m n (c p q) -> b c (m p) (n q)',
                      p=self.mar.patch_size, q=self.mar.patch_size)

        x = self.vae.decode(z)
        return x

    # LLM에 넣을 argument를 한 번에 준비하는 함수 : (input_embeds, attention_mask, position_ids, past_key_values)
    # past_key_values : True > 이미지 token 만 이어서 넣기
    # past_key_values : None > llm.get_input_embeddings()로 inputs_embeds 생성 + x concatenate = [text_embeds, image_embeds]
    def prepare_forward_input(self,
                              x,
                              inputs_embeds=None,
                              input_ids=None,
                              attention_mask=None,
                              past_key_values=None):
        b, l, _ = x.shape
        attention_mask = attention_mask.to(device=self.device, dtype=torch.bool)
        attention_mask = torch.cat([
            attention_mask, attention_mask.new_ones(b, l)
        ], dim=1)
        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids[position_ids < 0] = 0

        # import pdb; pdb.set_trace()

        # prepare context
        if past_key_values is not None:
            inputs_embeds = x
            position_ids = position_ids[:, -l:]
        else:
            if inputs_embeds is None:
                input_ids = input_ids.to(self.device)
                inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([inputs_embeds, x], dim=1)

        return dict(inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values)

    # null_embeds : MAR fake latent for masked tokens
    # forward_mae_encoder : 전체 sequence를 encode > x_enc
    # LLM 공간으로 projection 된 feature > z_enc
    def extract_visual_feature(self, x, mask=None, detach=False):
        b, m, n, _ = x.shape
        x = x.view(b, m*n, -1)
        # x: b mn c
        if mask is None:
            mask = torch.zeros_like(x[..., 0])
        null_embeds = self.mar.fake_latent.expand(x.shape[0], -1)
        x_enc = self.mar.forward_mae_encoder(x, mask, null_embeds, image_shape=(m, n))

        z_enc = self.proj_in(x_enc)
        # Move buffers to the end of the image sequence
        z_enc = torch.cat([
            z_enc[:, self.mar.buffer_size:],
            z_enc[:, :self.mar.buffer_size]], dim=1)

        if detach:
            x_enc = x_enc.detach()
            z_enc = z_enc.detach()

        return x_enc, z_enc

    def forward_mae_encoder(self, x, mask, detach=False, **context):
        # 현재 image token + text context를 LLM을 통해 섞어서 MAR encoder feature update
        b, m, n, _ = x.shape
        x_enc, z_enc = self.extract_visual_feature(x, mask=mask, detach=detach)
        inputs = self.prepare_forward_input(x=z_enc, **context)
        output = self.llm_model(**inputs, return_dict=True)

        z_llm = output.last_hidden_state[:, -z_enc.shape[1]:]
        # LLM forward 결과에서 마지막 z_enc 길이만큼 잘라서 z_llm(image 부분의 LLM hidden)만 뽑음

        # move buffers back to the start of the image sequence
        # buffer가 앞에 오도록 복원
        z_llm = torch.cat([
            z_llm[:, -self.mar.buffer_size:],
            z_llm[:, :-self.mar.buffer_size]], dim=1)

        # residual learning
        x_enc = x_enc + self.proj_out(z_llm)

        return x_enc

    # LLM KV cache 길이 맞추기 : 현재 context 길이(cur_len)까지만 남기고 나머지 자르기
    @staticmethod
    def curtail_cache(past_key_values, cur_len):
        for past_key_values_ in past_key_values:
            keys, values = past_key_values_
            keys.data = keys.data[:, :, :cur_len]
            values.data = values.data[:, :, :cur_len]

    @torch.no_grad()
    def prepare_text_conditions(self, prompt, cfg_prompt='Generate an image.'):
        all_prompts = [self.prompt_template['INSTRUCTION'].format(input=prompt),
                       self.prompt_template['INSTRUCTION'].format(input=cfg_prompt)]

        input_ids = [self.tokenizer.encode(p, add_special_tokens=True, return_tensors='pt')[0]
                     for p in all_prompts]
        valid_lens = [len(input_ids_) for input_ids_ in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True,
                                 padding_value=self.tokenizer.eos_token_id)
        attention_mask = torch.zeros_like(input_ids).bool()
        for i in range(len(input_ids)):
            attention_mask[i, :valid_lens[i]] = True

        return dict(input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device))

    @torch.no_grad()
    def prepare_past_key_values(self, input_ids=None, inputs_embeds=None):
        """Prepare prompt-only KV cache for the LLM.

        Runs a single LLM forward pass on the prompt embeddings to build `past_key_values`,
        so later steps can pass only image-token embeddings while reusing the prompt context.
        """
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either `input_ids` or `inputs_embeds` must be provided.")
            input_ids = input_ids.to(self.device)
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        output = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            position_ids=None,
            past_key_values=DynamicCache.from_legacy_cache(),
            return_dict=True,
            use_cache=True
        )
        return output.past_key_values

    @torch.no_grad()
    def sample_step_tokens(
        self,
        step: int,
        num_iter: int,
        *,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        tokens=None,
        mask=None,
        orders=None,
        past_key_values=None,
        cfg: float = 1.0,
        cfg_schedule: str = "constant",
        temperature: float = 1.0,
        image_shape=None,
        x_con=None,
        # --- NEW: external control (for global scheduling) ---
        mask_to_pred=None,   # (B,m,n) or (B,m*n) bool/0-1. If provided, bypass internal schedule.
        mask_next=None,      # (B,m,n) or (B,m*n) optional. If not provided, will set predicted positions to known(0).
    ):
        """Run exactly ONE masked-autoregressive sampling step and return updated tokens.

        Intended for multi-view / multi-tile global-merge loops:
        - gather global tokens -> per-view tokens
        - run one step per view
        - merge back to global
        - repeat for step=0..K-1

        Conventions:
        - mask==1 means "unknown / to-be-predicted"
        - tokens are continuous latents (no VQ) with last dim = self.token_embed_dim

        External-control mode:
        - If `mask_to_pred` is provided, the function will sample *only* those positions
            and will NOT compute cosine schedule / mask_by_order internally.
        - If `mask_next` is provided together, it will be returned as the updated `mask`.
            Otherwise, updated `mask` is computed as: mask[predicted_positions]=0 (known).

        Returns:
            dict: tokens/mask/mask_to_pred in (B, m, n, ...) layout.
        """
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        if attention_mask is None:
            raise ValueError("`attention_mask` must be provided for sampling.")

        bsz = attention_mask.shape[0]
        if cfg != 1.0:
            assert bsz % 2 == 0, "When cfg != 1.0, batch size must be even (cond/uncond pairs)."

        if image_shape is None:
            m = n = int(self.gen_seq_len ** 0.5)
        else:
            m, n = image_shape

        # tokens: (B, m*n, D)
        if tokens is None:
            tokens = torch.zeros(bsz, m * n, self.token_embed_dim, device=self.device, dtype=self.dtype)
        else:
            if tokens.dim() == 4:
                tokens = tokens.view(bsz, m * n, -1)
            tokens = tokens.to(device=self.device, dtype=self.dtype)
            if tokens.shape[-1] != self.token_embed_dim:
                raise ValueError(f"tokens last dim must be {self.token_embed_dim}, got {tokens.shape[-1]}")

        # mask: (B, m*n)
        if mask is None:
            mask = torch.ones(bsz, m * n, device=self.device, dtype=self.dtype)
        else:
            if mask.dim() == 3:
                mask = mask.view(bsz, m * n)
            mask = mask.to(device=self.device, dtype=self.dtype)

        # orders: (B, m*n)
        if orders is None:
            orders = self.mar.sample_orders(bsz, seq_len=m * n)
        else:
            orders = orders.view(bsz, m * n).to(device=self.device)

        if past_key_values is None:
            past_key_values = self.prepare_past_key_values(inputs_embeds=inputs_embeds)

        # keep cond/uncond aligned
        if cfg != 1.0:
            orders = orders.clone()
            orders[bsz // 2:] = orders[:bsz // 2]
            tokens = tokens.clone()
            tokens[bsz // 2:] = tokens[:bsz // 2]
            mask = mask.clone()
            mask[bsz // 2:] = mask[:bsz // 2]
            # external masks also need alignment
            if mask_to_pred is not None:
                mask_to_pred = mask_to_pred.clone()
                if mask_to_pred.shape[0] == bsz:
                    mask_to_pred[bsz // 2:] = mask_to_pred[:bsz // 2]
            if mask_next is not None:
                mask_next = mask_next.clone()
                if mask_next.shape[0] == bsz:
                    mask_next[bsz // 2:] = mask_next[:bsz // 2]

        cur_tokens = tokens.clone()

        # ---- encoder/decoder forward ----
        x_enc = self.forward_mae_encoder(
            tokens.view(bsz, m, n, -1),
            mask.to(self.dtype),
            past_key_values=past_key_values,
            attention_mask=attention_mask
        )
        if inputs_embeds is not None:
            self.curtail_cache(past_key_values, inputs_embeds.shape[1])

        z = self.mar.forward_mae_decoder(
            x_enc,
            mask.to(self.dtype),
            image_shape=(m, n),
            x_con=x_con
        )
        # expected z shape aligns with (B, m*n, Dz) for indexing by (batch_idx, token_idx)

        # -------------------------------------------------------------------------
        # Mask schedule / mask_to_pred selection
        # -------------------------------------------------------------------------
        if mask_to_pred is not None:
            # external-control mode: caller decides which positions to predict this step
            if mask_to_pred.dim() == 3:
                mask_to_pred = mask_to_pred.view(bsz, m * n)
            # accept bool or 0/1 float
            if mask_to_pred.dtype != torch.bool:
                mask_to_pred = (mask_to_pred > 0.5)
            mask_to_pred = mask_to_pred.to(device=self.device)

            # updated mask for next step
            if mask_next is not None:
                if mask_next.dim() == 3:
                    mask_next = mask_next.view(bsz, m * n)
                mask_next = mask_next.to(device=self.device, dtype=self.dtype)
                mask = mask_next
            else:
                # default: mark predicted positions as known(0), keep others as-is
                mask = mask.clone()
                mask[mask_to_pred] = 0.0

            # for CFG, enforce identical prediction locations across cond/uncond
            if cfg != 1.0:
                mask_to_pred = mask_to_pred.clone()
                mask_to_pred[bsz // 2:] = mask_to_pred[:bsz // 2]
                mask = mask.clone()
                mask[bsz // 2:] = mask[:bsz // 2]

            # cfg_iter schedule in external-control mode:
            # we approximate progress by current known ratio (optional); simplest is constant/linear by remaining unknown count.
            unknown_cnt = torch.sum(mask, dim=-1, keepdim=False)  # (B,)
            # use first element as representative (cond/uncond aligned)
            unknown_cnt0 = unknown_cnt[0].detach()
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (m * n - unknown_cnt0) / (m * n)
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError(f"Unknown cfg_schedule: {cfg_schedule}")

        else:
            # original internal cosine schedule
            mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / num_iter)
            mask_len = torch.tensor([np.floor(m * n * mask_ratio)], device=self.device)

            # masks out at least one for the next iteration
            mask_len = torch.maximum(
                torch.tensor([1.0], device=self.device),
                torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1.0, mask_len)
            )

            mask_next = mask_by_order(mask_len[0], orders, bsz, m * n).to(self.device)
            if cfg != 1.0:
                mask_next[bsz // 2:] = mask_next[:bsz // 2]

            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())

            mask = mask_next

            # cfg schedule (original)
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (m * n - mask_len[0]) / (m * n)
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError(f"Unknown cfg_schedule: {cfg_schedule}")

        # -------------------------------------------------------------------------
        # Sample token latents for this step and write back
        # -------------------------------------------------------------------------
        idx = mask_to_pred.nonzero(as_tuple=True)  # (batch_idx, token_idx)
        z_sel = z[idx]  # (N_pred, Dz)

        sampled_token_latent = self.mar.diffloss.sample(
            z_sel, temperature, cfg_iter
        ).to(self.dtype)

        cur_tokens[idx] = sampled_token_latent

        if cfg != 1.0:
            cur_tokens[bsz // 2:] = cur_tokens[:bsz // 2]

        return {
            "tokens": cur_tokens.view(bsz, m, n, -1),
            "mask": mask.view(bsz, m, n),
            "mask_to_pred": mask_to_pred.view(bsz, m, n),
            "past_key_values": past_key_values,
            "orders": orders,
        }

    # @torch.no_grad()
    # def sample_step_tokens(
    #     self,
    #     step: int,
    #     num_iter: int,
    #     *,
    #     input_ids=None,
    #     inputs_embeds=None,
    #     attention_mask=None,
    #     tokens=None,
    #     mask=None,
    #     orders=None,
    #     past_key_values=None,
    #     cfg: float = 1.0,
    #     cfg_schedule: str = "constant",
    #     temperature: float = 1.0,
    #     image_shape=None,
    #     x_con=None,
    # ):
    #     """Run exactly ONE masked-autoregressive sampling step and return updated tokens.

    #     Intended for multi-view / multi-tile global-merge loops:
    #       - gather global tokens -> per-view tokens
    #       - run one step per view
    #       - merge back to global
    #       - repeat for step=0..K-1

    #     Conventions:
    #       - mask==1 means "unknown / to-be-predicted"
    #       - tokens are continuous latents (no VQ) with last dim = self.token_embed_dim

    #     Returns:
    #         dict: tokens/mask/mask_to_pred in (B, m, n, ...) layout.
    #     """
    #     if inputs_embeds is None and input_ids is not None:
    #         inputs_embeds = self.llm.get_input_embeddings()(input_ids)

    #     if attention_mask is None:
    #         raise ValueError("`attention_mask` must be provided for sampling.")

    #     bsz = attention_mask.shape[0]
    #     if cfg != 1.0:
    #         assert bsz % 2 == 0, "When cfg != 1.0, batch size must be even (cond/uncond pairs)."

    #     if image_shape is None:
    #         m = n = int(self.gen_seq_len ** 0.5)
    #     else:
    #         m, n = image_shape

    #     # tokens: (B, m*n, D)
    #     if tokens is None:
    #         tokens = torch.zeros(bsz, m * n, self.token_embed_dim, device=self.device, dtype=self.dtype)
    #     else:
    #         if tokens.dim() == 4:
    #             tokens = tokens.view(bsz, m * n, -1)
    #         tokens = tokens.to(device=self.device, dtype=self.dtype)
    #         if tokens.shape[-1] != self.token_embed_dim:
    #             raise ValueError(f"tokens last dim must be {self.token_embed_dim}, got {tokens.shape[-1]}")

    #     # mask: (B, m*n)
    #     if mask is None:
    #         mask = torch.ones(bsz, m * n, device=self.device, dtype=self.dtype)
    #     else:
    #         if mask.dim() == 3:
    #             mask = mask.view(bsz, m * n)
    #         mask = mask.to(device=self.device, dtype=self.dtype)

    #     # orders: (B, m*n)
    #     if orders is None:
    #         orders = self.mar.sample_orders(bsz, seq_len=m * n)
    #     else:
    #         orders = orders.view(bsz, m * n).to(device=self.device)

    #     if past_key_values is None:
    #         past_key_values = self.prepare_past_key_values(inputs_embeds=inputs_embeds)

    #     # keep cond/uncond aligned
    #     if cfg != 1.0:
    #         orders = orders.clone()
    #         orders[bsz // 2:] = orders[:bsz // 2]
    #         tokens = tokens.clone()
    #         tokens[bsz // 2:] = tokens[:bsz // 2]
    #         mask = mask.clone()
    #         mask[bsz // 2:] = mask[:bsz // 2]

    #     cur_tokens = tokens.clone()

    #     x_enc = self.forward_mae_encoder(
    #         tokens.view(bsz, m, n, -1),
    #         mask.to(self.dtype),
    #         past_key_values=past_key_values,
    #         attention_mask=attention_mask
    #     )
    #     if inputs_embeds is not None:
    #         self.curtail_cache(past_key_values, inputs_embeds.shape[1])

    #     z = self.mar.forward_mae_decoder(x_enc, mask.to(self.dtype), image_shape=(m, n), x_con=x_con)

    #     # cosine mask schedule
    #     mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
    #     mask_len = torch.Tensor([np.floor(m * n * mask_ratio)]).to(self.device)

    #     # masks out at least one for the next iteration
    #     mask_len = torch.maximum(
    #         torch.Tensor([1]).to(self.device),
    #         torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len)
    #     )

    #     mask_next = mask_by_order(mask_len[0], orders, bsz, m * n).to(self.device)
    #     if cfg != 1.0:
    #         mask_next[bsz // 2:] = mask_next[:bsz // 2]

    #     if step >= num_iter - 1:
    #         mask_to_pred = mask[:bsz].bool()
    #     else:
    #         mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())

    #     mask = mask_next

    #     # sample token latents for this step
    #     z = z[mask_to_pred.nonzero(as_tuple=True)]
    #     if cfg_schedule == "linear":
    #         cfg_iter = 1 + (cfg - 1) * (m * n - mask_len[0]) / (m * n)
    #     elif cfg_schedule == "constant":
    #         cfg_iter = cfg
    #     else:
    #         raise NotImplementedError

    #     sampled_token_latent = self.mar.diffloss.sample(z, temperature, cfg_iter).to(self.dtype)
    #     cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent

    #     if cfg != 1.0:
    #         cur_tokens[bsz // 2:] = cur_tokens[:bsz // 2]

    #     return {
    #         "tokens": cur_tokens.view(bsz, m, n, -1),
    #         "mask": mask.view(bsz, m, n),
    #         "mask_to_pred": mask_to_pred.view(bsz, m, n),
    #         "past_key_values": past_key_values,
    #         "orders": orders,
    #     }