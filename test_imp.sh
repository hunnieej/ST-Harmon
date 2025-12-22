#!/usr/bin/env bash
set -euo pipefail

# ====== User-configurable defaults (env로 덮어쓰기 가능) ======
GPU_ID="${GPU_ID:-1}"
CONFIG_REL="${CONFIG_REL:-configs/models/qwen2_5_1_5b_kl16_mar_p.py}"
CKPT_REL="${CKPT_REL:-checkpoints/harmon_1.5b.pth}"
NUM_TILES="${NUM_TILES:-9}"

PROMPTS_FILE="${1:-prompt.txt}"  # HARMON 루트 기준 prompt.txt

# ====== Resolve HARMON root robustly ======
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/scripts/text2pano_temp.py" ]]; then
  ROOT="${SCRIPT_DIR}"                         # sh가 HARMON 루트에 있는 경우
elif [[ -f "${SCRIPT_DIR}/../scripts/text2pano_temp.py" ]]; then
  ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"       # sh가 HARMON/scripts 안에 있는 경우
elif git -C "${SCRIPT_DIR}" rev-parse --show-toplevel >/dev/null 2>&1; then
  ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"  # git repo면 최상단
else
  echo "[ERROR] Cannot locate HARMON root (scripts/text2pano_temp.py not found)."
  echo "        Place this .sh in HARMON root or HARMON/scripts."
  exit 1
fi

# ====== Python binary ======
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
else
  PY="${PYTHON_BIN:-python3}"
fi

# ====== Paths ======
CONFIG="${ROOT}/${CONFIG_REL}"
CKPT="${ROOT}/${CKPT_REL}"
TEXT2PANO="${ROOT}/scripts/text2pano_temp.py"

# ====== Output base: Tests/YYMMDD_ver2 ======
DATE_TAG="$(date +%y%m%d)"
BASE_OUTDIR="${BASE_OUTDIR:-${ROOT}/Tests/${DATE_TAG}_imp}"
mkdir -p "${BASE_OUTDIR}"

export PYTHONPATH="${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# ====== Prompt file resolve (relative -> ROOT) ======
if [[ ! -f "${PROMPTS_FILE}" ]]; then
  if [[ -f "${ROOT}/${PROMPTS_FILE}" ]]; then
    PROMPTS_FILE="${ROOT}/${PROMPTS_FILE}"
  else
    echo "[ERROR] prompt file not found: ${PROMPTS_FILE}"
    echo "        (also tried: ${ROOT}/${PROMPTS_FILE})"
    exit 1
  fi
fi

# ====== Sanity checks ======
[[ -f "${CONFIG}" ]] || { echo "[ERROR] config not found: ${CONFIG}"; exit 1; }
[[ -f "${CKPT}"   ]] || { echo "[ERROR] checkpoint not found: ${CKPT}"; exit 1; }
[[ -f "${TEXT2PANO}" ]] || { echo "[ERROR] script not found: ${TEXT2PANO}"; exit 1; }

echo "[INFO] ROOT=${ROOT}"
echo "[INFO] PROMPTS_FILE=${PROMPTS_FILE}"
echo "[INFO] OUTDIR=${BASE_OUTDIR}"
echo "[INFO] GPU=${GPU_ID}"
echo "[INFO] CONFIG=${CONFIG}"
echo "[INFO] CKPT=${CKPT}"
echo "[INFO] NUM_TILES=${NUM_TILES}"
echo

i=0
while IFS= read -r prompt || [[ -n "${prompt}" ]]; do
  # 빈 줄/주석(#로 시작) 스킵
  [[ "${prompt}" =~ ^[[:space:]]*$ ]] && continue
  [[ "${prompt}" =~ ^[[:space:]]*# ]] && continue

  i=$((i+1))
  test_dir="${BASE_OUTDIR}/test_$(printf "%03d" "${i}")"
  mkdir -p "${test_dir}"

  out_img="${test_dir}/output.jpg"
  printf "%s\n" "${prompt}" > "${test_dir}/prompt.txt"

  echo "[RUN] test_$(printf "%03d" "${i}") -> ${out_img}"

  "${PY}" "${TEXT2PANO}" \
    "${CONFIG}" \
    --checkpoint "${CKPT}" \
    --prompt "${prompt}" \
    --output "${out_img}" \
    --num_tiles "${NUM_TILES}" \
    2>&1 | tee "${test_dir}/stdout.log"

done < "${PROMPTS_FILE}"

echo
echo "[DONE] Saved under: ${BASE_OUTDIR}"
