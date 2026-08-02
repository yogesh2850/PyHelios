"""Phase 9 / T9.1 — PyHelios geometry -> OpenUSD (.usda) exporter (texture-accurate).

Runs in the `helios` conda env (no `pxr` there). A USDA file is plain ASCII, so we emit a
`UsdGeomMesh` — now with a real `UsdPreviewSurface` material network — directly from Helios
primitive geometry, no USD library needed on the producing side. The Isaac Sim side
(`open_in_isaacsim.py`) parses it with the real `pxr` runtime, so the format is validated
by an independent consumer.

Texture-accurate bridge (the Phase 9 texture fix):
  Helios apple primitives are texture-mapped, not solid-coloured — `getPrimitiveColor`
  returns (0,0,0) for them, which is why the old bridge fell back to a cosmetic tint. The
  real signal lives in `getPrimitiveTextureFile(uuid)` (which image) + `getPrimitiveTextureUV`
  (per-vertex UVs). This exporter reads both and authors, per distinct texture:
    * a UsdPreviewSurface bound to that image via UsdUVTexture + a `st` UsdPrimvarReader,
    * a per-vertex `primvars:st` (faceVarying) texCoord2f array,
    * a GeomSubset (family "materialBind") grouping the faces that use it.
  AppleLeaf.png carries an alpha cutout, so the leaf material wires the texture alpha into
  `opacity` with an `opacityThreshold` — leaves render as real leaf silhouettes, not green
  quads. Primitives with no texture (petioles/shoots) keep their solid displayColor and are
  left unbound (RTX shades them from displayColor). Textures are copied next to the .usda and
  referenced relatively so the stage is portable.

Coordinate/units: Helios is Z-up, metres; USD carries that natively (upAxis="Z",
metersPerUnit=1), so geometry crosses 1:1 with no vertex transform.
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

from pyhelios import Context, PlantArchitecture
from pyhelios.types import vec3


# ---------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------
def build_tree(context, plantarch, age_days, seed, base_position=(0.0, 0.0, 0.0)):
    """Build one apple tree at `base_position` on a seeded Context. Returns its UUIDs."""
    context.seedRandomGenerator(seed)  # determinism (Phase 2/5/8 finding)
    plantarch.loadPlantModelFromLibrary("apple")
    plantarch.setCollisionRelevantOrgans(
        include_internodes=True, include_leaves=True, include_fruit=True
    )
    plantarch.enableSoftCollisionAvoidance(enable_fruit_collision=True)
    pid = plantarch.buildPlantInstanceFromLibrary(
        base_position=vec3(*base_position), age=age_days, build_parameters=None
    )
    return plantarch.getAllPlantUUIDs(pid)


def _fallback_color(nx, ny, nz):
    """Legibility tint for the few untextured, reported-black primitives (petioles etc.)."""
    if abs(nz) > 0.85:
        return (0.32, 0.20, 0.11)  # bark brown
    return (0.20, 0.45, 0.16)  # leaf green


# ---------------------------------------------------------------------------------------
# Extract (batch APIs — a fully-grown tree is ~270k primitives)
# ---------------------------------------------------------------------------------------
def extract_mesh(context, uuids):
    """Pull Helios geometry + per-vertex UVs + per-primitive texture into USD-ready arrays.

    Returns a dict with flat point/uv/color arrays, faceVertexCounts/Indices, and a mapping
    texture-file -> list of face indices (for GeomSubsets). Untextured faces go under the key
    "" and are shaded from displayColor. Quads (4 verts) are split 0-1-2 / 0-2-3, and their
    UVs are split the same way so texture coords stay consistent with the triangulation.
    """
    n = len(uuids)
    vflat, voff = context.getPrimitiveVertices(uuids)  # float32, uint32 offsets [N+1]
    uvflat, uvoff = context.getPrimitiveTextureUV(uuids)  # float32, uint32 offsets [N+1]
    texfiles = context.getPrimitiveTextureFile(uuids)  # list[str], len N
    colors = context.getPrimitiveColor(uuids)  # (N,3) float32

    points = []          # (x,y,z) per mesh vertex, non-indexed
    st = []              # (u,v) per mesh vertex, aligned with points
    disp = []            # (r,g,b) per face (uniform)
    face_counts = []     # all 3
    faces_by_tex = {}    # texture path -> [face_index, ...]
    stats = {
        "primitives": n, "tris_from_tri": 0, "tris_from_quad": 0,
        "skipped": 0, "textured_faces": 0, "untextured_faces": 0,
        "vertex_hist": {}, "faces_per_texture": {},
    }

    face_idx = 0
    for i in range(n):
        a, b = int(voff[i]), int(voff[i + 1])
        verts = vflat[a:b].reshape(-1, 3)
        nv = verts.shape[0]
        stats["vertex_hist"][nv] = stats["vertex_hist"].get(nv, 0) + 1

        ua, ub = int(uvoff[i]), int(uvoff[i + 1])
        uvs = uvflat[ua:ub].reshape(-1, 2) if ub > ua else None

        if nv == 3:
            tri_sets = [(0, 1, 2)]
            stats["tris_from_tri"] += 1
        elif nv == 4:
            tri_sets = [(0, 1, 2), (0, 2, 3)]
            stats["tris_from_quad"] += 1
        else:
            stats["skipped"] += 1
            continue

        tex = texfiles[i] or ""
        has_uv = uvs is not None and uvs.shape[0] == nv
        if not has_uv:
            tex = ""  # can't texture without matching UVs — treat as solid

        if tex:
            r = g = b = 1.0  # texture supplies colour; keep displayColor neutral
            stats["textured_faces"] += len(tri_sets)
        else:
            r, g, bb = float(colors[i][0]), float(colors[i][1]), float(colors[i][2])
            if r == 0.0 and g == 0.0 and bb == 0.0:
                nrm = context.getPrimitiveNormal(int(uuids[i]))
                r, g, bb = _fallback_color(float(nrm.x), float(nrm.y), float(nrm.z))
            b = bb
            stats["untextured_faces"] += len(tri_sets)

        for tri in tri_sets:
            for k in tri:
                vx, vy, vz = verts[k]
                points.append((float(vx), float(vy), float(vz)))
                if has_uv:
                    st.append((float(uvs[k][0]), float(uvs[k][1])))
                else:
                    st.append((0.0, 0.0))
            face_counts.append(3)
            disp.append((r, g, b))
            faces_by_tex.setdefault(tex, []).append(face_idx)
            face_idx += 1

    for tex, fl in faces_by_tex.items():
        key = os.path.basename(tex) if tex else "(untextured)"
        stats["faces_per_texture"][key] = len(fl)

    indices = list(range(len(points)))
    return {
        "points": points, "st": st, "disp": disp,
        "face_counts": face_counts, "indices": indices,
        "faces_by_tex": faces_by_tex, "stats": stats,
    }


# ---------------------------------------------------------------------------------------
# USD authoring (ASCII, no pxr)
# ---------------------------------------------------------------------------------------
def _sanitize(name):
    return "".join(c if c.isalnum() else "_" for c in name)


def _fmt_pts(arr):
    return ", ".join(f"({x:.6g}, {y:.6g}, {z:.6g})" for x, y, z in arr)


def _fmt_uv(arr):
    return ", ".join(f"({u:.6g}, {v:.6g})" for u, v in arr)


def _fmt_rgb(arr):
    return ", ".join(f"({r:.4g}, {g:.4g}, {b:.4g})" for r, g, b in arr)


def _material_block(root, mat_name, rel_tex, is_cutout, indent="        "):
    """Author a UsdPreviewSurface + UsdUVTexture + st reader network for one texture."""
    base = f"{root}/Looks/{mat_name}"
    L = []
    ap = L.append
    ap(f'{indent}def Material "{mat_name}"')
    ap(f"{indent}{{")
    ap(f"{indent}    token outputs:surface.connect = <{base}/Surface.outputs:surface>")
    ap(f'{indent}    def Shader "Surface"')
    ap(f"{indent}    {{")
    ap(f'{indent}        uniform token info:id = "UsdPreviewSurface"')
    ap(f"{indent}        color3f inputs:diffuseColor.connect = <{base}/Tex.outputs:rgb>")
    ap(f"{indent}        float inputs:roughness = 0.85")
    ap(f"{indent}        float inputs:metallic = 0")
    if is_cutout:
        ap(f"{indent}        float inputs:opacity.connect = <{base}/Tex.outputs:a>")
        ap(f"{indent}        float inputs:opacityThreshold = 0.5")
    ap(f"{indent}        token outputs:surface")
    ap(f"{indent}    }}")
    ap(f'{indent}    def Shader "Tex"')
    ap(f"{indent}    {{")
    ap(f'{indent}        uniform token info:id = "UsdUVTexture"')
    ap(f"{indent}        asset inputs:file = @{rel_tex}@")
    ap(f"{indent}        float2 inputs:st.connect = <{base}/stReader.outputs:result>")
    ap(f'{indent}        token inputs:wrapS = "repeat"')
    ap(f'{indent}        token inputs:wrapT = "repeat"')
    ap(f"{indent}        float3 outputs:rgb")
    if is_cutout:
        ap(f"{indent}        float outputs:a")
    ap(f"{indent}    }}")
    ap(f'{indent}    def Shader "stReader"')
    ap(f"{indent}    {{")
    ap(f'{indent}        uniform token info:id = "UsdPrimvarReader_float2"')
    ap(f'{indent}        token inputs:varname = "st"')
    ap(f"{indent}        float2 outputs:result")
    ap(f"{indent}    }}")
    ap(f"{indent}}}")
    return "\n".join(L)


def write_tree_usda(path, mesh, tex_dir_rel, texture_srcs, root_prim="AppleTree"):
    """Write a self-contained textured tree stage. `texture_srcs`: {texpath: dest_basename}."""
    root = f"/{root_prim}"
    # Assign each distinct texture a material name; build subset -> material bindings.
    tex_to_mat = {}
    cutout = {}
    mat_defs = []
    for tex in mesh["faces_by_tex"]:
        if not tex:
            continue
        mat = _sanitize(os.path.splitext(os.path.basename(tex))[0])
        tex_to_mat[tex] = mat
        is_cut = os.path.basename(tex).lower() == "appleleaf.png"
        cutout[tex] = is_cut
        rel = f"./{tex_dir_rel}/{texture_srcs[tex]}"
        mat_defs.append(_material_block(root, mat, rel, is_cut))

    with open(path, "w") as f:
        w = f.write
        w("#usda 1.0\n(\n")
        w(f'    defaultPrim = "{root_prim}"\n')
        w("    metersPerUnit = 1\n")
        w('    upAxis = "Z"\n)\n\n')
        w(f'def Xform "{root_prim}"\n{{\n')

        # Materials
        w('    def Scope "Looks"\n    {\n')
        w("\n".join(mat_defs))
        w("\n    }\n\n")

        # Mesh
        w('    def Mesh "Geom"\n    {\n')
        w("        int[] faceVertexCounts = ["
          + ", ".join(str(c) for c in mesh["face_counts"]) + "]\n")
        w("        int[] faceVertexIndices = ["
          + ", ".join(str(i) for i in mesh["indices"]) + "]\n")
        w("        point3f[] points = [" + _fmt_pts(mesh["points"]) + "]\n")
        w("        texCoord2f[] primvars:st = [" + _fmt_uv(mesh["st"]) + "] (\n")
        w('            interpolation = "faceVarying"\n        )\n')
        w("        color3f[] primvars:displayColor = [" + _fmt_rgb(mesh["disp"]) + "] (\n")
        w('            interpolation = "uniform"\n        )\n')
        w('        uniform token subdivisionScheme = "none"\n')

        # GeomSubsets bound to materials (untextured faces stay unbound -> displayColor)
        for tex, faces in mesh["faces_by_tex"].items():
            if not tex:
                continue
            mat = tex_to_mat[tex]
            sub = f"subset_{mat}"
            w(f'        def GeomSubset "{sub}"\n        {{\n')
            w('            uniform token elementType = "face"\n')
            w('            uniform token familyName = "materialBind"\n')
            w("            int[] indices = [" + ", ".join(str(i) for i in faces) + "]\n")
            w(f"            rel material:binding = <{root}/Looks/{mat}>\n")
            w("        }\n")

        w("    }\n")
        w("}\n")


def copy_textures(mesh, out_dir):
    """Copy every distinct texture next to the stage under textures/. Returns {src: base}."""
    tex_dir = os.path.join(out_dir, "textures")
    os.makedirs(tex_dir, exist_ok=True)
    srcs = {}
    for tex in mesh["faces_by_tex"]:
        if not tex:
            continue
        base = os.path.basename(tex)
        dst = os.path.join(tex_dir, base)
        if not os.path.exists(dst):
            shutil.copyfile(tex, dst)
        srcs[tex] = base
    return srcs


# ---------------------------------------------------------------------------------------
# CLI: single textured tree
# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--age-days", type=float, default=1825.0)  # fully grown
    ap.add_argument("--seed", type=int, default=9001)
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "output", "apple_tree.usda"),
    )
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    with Context() as context:
        with PlantArchitecture(context) as plantarch:
            uuids = build_tree(context, plantarch, args.age_days, args.seed)
        build_s = time.time() - t0

        t1 = time.time()
        mesh = extract_mesh(context, uuids)
        extract_s = time.time() - t1

        xb, yb, zb = context.getDomainBoundingBox(uuids)
        bbox = {"x": [float(xb.x), float(xb.y)], "y": [float(yb.x), float(yb.y)],
                "z": [float(zb.x), float(zb.y)]}

    t2 = time.time()
    tex_srcs = copy_textures(mesh, out_dir)
    write_tree_usda(args.out, mesh, "textures", tex_srcs)
    write_s = time.time() - t2
    size_mb = os.path.getsize(args.out) / 1e6

    report = {
        "out": args.out, "size_mb": round(size_mb, 3),
        "age_days": args.age_days, "seed": args.seed,
        "num_points": len(mesh["points"]), "num_faces": len(mesh["face_counts"]),
        "textures": [os.path.basename(t) for t in tex_srcs],
        "bbox_m": bbox,
        "timing_s": {"build": round(build_s, 2), "extract": round(extract_s, 2),
                     "write": round(write_s, 2)},
        "geometry_stats": mesh["stats"],
    }
    report_path = os.path.splitext(args.out)[0] + "_export_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nWrote USD: {args.out}\nWrote report: {report_path}")


if __name__ == "__main__":
    sys.exit(main())
