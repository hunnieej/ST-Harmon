import argparse
import json
import math
import struct
from pathlib import Path

# glTF constants
GL_ARRAY_BUFFER = 34962
GL_ELEMENT_ARRAY_BUFFER = 34963

GL_FLOAT = 5126
GL_UNSIGNED_SHORT = 5123
GL_UNSIGNED_INT = 5125

WRAP_REPEAT = 10497
MAG_LINEAR = 9729
MIN_LINEAR = 9729

def pad4_len(n: int) -> int:
    return (n + 3) & ~3

def pad4(b: bytes) -> bytes:
    return b + b"\x00" * ((4 - (len(b) % 4)) % 4)

def pack_f32(arr):
    return struct.pack("<" + "f" * len(arr), *arr)

def pack_u16(arr):
    return struct.pack("<" + "H" * len(arr), *arr)

def pack_u32(arr):
    return struct.pack("<" + "I" * len(arr), *arr)

def guess_mime(tex_path: Path) -> str:
    suf = tex_path.suffix.lower()
    if suf == ".png":
        return "image/png"
    if suf in [".jpg", ".jpeg"]:
        return "image/jpeg"
    # glTF 2.0 images should specify mimeType for bufferView images
    raise ValueError(f"Unsupported texture extension: {tex_path.suffix} (use .png/.jpg)")

def make_cylinder_mesh(radius: float, height: float, seg_u: int, seg_v: int, inside: bool, flip_v: bool, flip_u: bool):
    """
    Cylinder aligned with Y axis (side surface only).
    Vert grid: (seg_v+1) x (seg_u+1) with duplicated seam column.
    UV:
      u in [0..1] around circumference
      v in [0..1] along height
    """
    verts = []
    norms = []
    uvs = []

    for j in range(seg_v + 1):
        v = j / seg_v
        y = (v - 0.5) * height

        v_tex = (1.0 - v) if flip_v else v

        for i in range(seg_u + 1):
            u = i / seg_u
            if flip_u:
                u_tex = 1.0 - u
            else:
                u_tex = u

            theta = u * 2.0 * math.pi

            x = radius * math.sin(theta)
            z = radius * math.cos(theta)

            # outward normal
            nx = math.sin(theta)
            nz = math.cos(theta)
            ny = 0.0
            if inside:
                nx, ny, nz = -nx, -ny, -nz

            verts.extend([x, y, z])
            norms.extend([nx, ny, nz])
            uvs.extend([u_tex, v_tex])

    idx = []
    cols = seg_u + 1
    for j in range(seg_v):
        for i in range(seg_u):
            a = j * cols + i
            b = a + 1
            c = (j + 1) * cols + i
            d = c + 1

            if not inside:
                idx.extend([a, c, b])
                idx.extend([b, c, d])
            else:
                # flip winding for inside
                idx.extend([a, b, c])
                idx.extend([b, d, c])

    vcount = (seg_v + 1) * (seg_u + 1)
    return verts, norms, uvs, idx, vcount

def compute_min_max_vec3(verts):
    xs = verts[0::3]; ys = verts[1::3]; zs = verts[2::3]
    return [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]

def write_glb(out_glb: Path, gltf_json: dict, bin_blob: bytes):
    # GLB header + chunks
    json_bytes = json.dumps(gltf_json, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    json_bytes = pad4(json_bytes)

    bin_blob = pad4(bin_blob)

    # GLB layout
    # header: 12 bytes
    # JSON chunk: 8 + len(json_bytes)
    # BIN chunk: 8 + len(bin_blob)
    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_blob)

    with open(out_glb, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total_len))
        f.write(struct.pack("<I4s", len(json_bytes), b"JSON"))
        f.write(json_bytes)
        f.write(struct.pack("<I4s", len(bin_blob), b"BIN\0"))
        f.write(bin_blob)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texture", type=str, required=True, help="Path to output.png/jpg (unwrapped cylinder texture).")
    ap.add_argument("--out", type=str, required=True, help="Output .glb path (single file).")
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--height", type=float, default=2.0)
    ap.add_argument("--seg_u", type=int, default=256)
    ap.add_argument("--seg_v", type=int, default=128)
    ap.add_argument("--inside", action="store_true", help="Texture on inside surface (panorama viewer style).")
    ap.add_argument("--flip_v", action="store_true", help="Flip V if upside down.")
    ap.add_argument("--flip_u", action="store_true", help="Flip U if left-right reversed.")
    ap.add_argument("--no_unlit", action="store_true", help="Disable KHR_materials_unlit (not recommended).")

    # emissive fallback always on (solves black even if unlit ignored)
    ap.add_argument("--emissive", action="store_true", default=True, help="Use emissiveTexture fallback (recommended).")
    ap.add_argument("--no_emissive", dest="emissive", action="store_false")

    args = ap.parse_args()

    tex_path = Path(args.texture).expanduser().resolve()
    out_glb = Path(args.out).expanduser().resolve()
    out_glb.parent.mkdir(parents=True, exist_ok=True)

    if not tex_path.exists():
        raise FileNotFoundError(tex_path)

    mime = guess_mime(tex_path)
    image_bytes = tex_path.read_bytes()

    verts, norms, uvs, idx, vcount = make_cylinder_mesh(
        radius=args.radius,
        height=args.height,
        seg_u=args.seg_u,
        seg_v=args.seg_v,
        inside=args.inside,
        flip_v=args.flip_v,
        flip_u=args.flip_u,
    )

    pos_min, pos_max = compute_min_max_vec3(verts)

    # Decide index type
    use_u32 = vcount > 65535
    if use_u32:
        idx_bytes = pack_u32(idx)
        idx_component = GL_UNSIGNED_INT
    else:
        idx_bytes = pack_u16(idx)
        idx_component = GL_UNSIGNED_SHORT

    # Pack BIN blob with: positions, normals, uvs, indices, image
    pos_bytes = pad4(pack_f32(verts))
    nor_bytes = pad4(pack_f32(norms))
    uv_bytes  = pad4(pack_f32(uvs))
    idx_bytes = pad4(idx_bytes)
    img_bytes = pad4(image_bytes)

    pos_off = 0
    nor_off = pos_off + len(pos_bytes)
    uv_off  = nor_off + len(nor_bytes)
    idx_off = uv_off  + len(uv_bytes)
    img_off = idx_off + len(idx_bytes)

    bin_blob = pos_bytes + nor_bytes + uv_bytes + idx_bytes + img_bytes

    # BufferViews: positions/normals/uv/indices/image
    bv_pos = 0
    bv_nor = 1
    bv_uv  = 2
    bv_idx = 3
    bv_img = 4

    acc_pos = 0
    acc_nor = 1
    acc_uv  = 2
    acc_idx = 3

    # Material: Unlit + Emissive fallback
    material = {
        "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0},
            "metallicFactor": 0.0,
            "roughnessFactor": 1.0
        },
        "doubleSided": True
    }

    extensionsUsed = []
    if not args.no_unlit:
        extensionsUsed.append("KHR_materials_unlit")
        material["extensions"] = {"KHR_materials_unlit": {}}

    if args.emissive:
        # Even if viewer ignores unlit and has no lights, emissive will show.
        material["emissiveTexture"] = {"index": 0}
        material["emissiveFactor"] = [1.0, 1.0, 1.0]

    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_off, "byteLength": len(pos_bytes), "target": GL_ARRAY_BUFFER},
            {"buffer": 0, "byteOffset": nor_off, "byteLength": len(nor_bytes), "target": GL_ARRAY_BUFFER},
            {"buffer": 0, "byteOffset": uv_off,  "byteLength": len(uv_bytes),  "target": GL_ARRAY_BUFFER},
            {"buffer": 0, "byteOffset": idx_off, "byteLength": len(idx_bytes), "target": GL_ELEMENT_ARRAY_BUFFER},
            # image in BIN
            {"buffer": 0, "byteOffset": img_off, "byteLength": len(img_bytes)},
        ],
        "accessors": [
            {"bufferView": bv_pos, "byteOffset": 0, "componentType": GL_FLOAT, "count": vcount, "type": "VEC3",
             "min": pos_min, "max": pos_max},
            {"bufferView": bv_nor, "byteOffset": 0, "componentType": GL_FLOAT, "count": vcount, "type": "VEC3"},
            {"bufferView": bv_uv,  "byteOffset": 0, "componentType": GL_FLOAT, "count": vcount, "type": "VEC2"},
            {"bufferView": bv_idx, "byteOffset": 0, "componentType": idx_component, "count": len(idx), "type": "SCALAR"},
        ],
        "images": [
            {"bufferView": bv_img, "mimeType": mime}
        ],
        "samplers": [
            {"magFilter": MAG_LINEAR, "minFilter": MIN_LINEAR, "wrapS": WRAP_REPEAT, "wrapT": WRAP_REPEAT}
        ],
        "textures": [
            {"sampler": 0, "source": 0}
        ],
        "materials": [material],
        "meshes": [
            {"primitives": [{
                "attributes": {"POSITION": acc_pos, "NORMAL": acc_nor, "TEXCOORD_0": acc_uv},
                "indices": acc_idx,
                "material": 0
            }]}
        ],
        "nodes": [{"mesh": 0, "name": "Cylinder"}],
        "scenes": [{"nodes": [0]}],
        "scene": 0
    }

    if extensionsUsed:
        gltf["extensionsUsed"] = extensionsUsed

    write_glb(out_glb, gltf, bin_blob)
    print(f"[OK] Wrote GLB (embedded texture+buffer): {out_glb}")
    print(f"[INFO] inside={args.inside}, flip_u={args.flip_u}, flip_v={args.flip_v}, unlit={not args.no_unlit}, emissive={args.emissive}")
    if use_u32:
        print("[INFO] Used uint32 indices (vertex count > 65535).")

if __name__ == "__main__":
    main()
