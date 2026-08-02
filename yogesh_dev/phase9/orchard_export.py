"""Phase 9 / T9.2b — a 45-tree apple orchard as an instanced OpenUSD stage.

Fully-grown apple trees are ~270k primitives each; 45 of them written as raw meshes would be
a multi-GB ASCII file that no viewer opens comfortably. Real orchards are one cultivar of
near-identical trees, so the correct USD idiom is *instancing*: author a small set of unique
trees once, then reference each one `instanceable = true` at every orchard position. GPU
memory and file size scale with the unique trees, not the 45 placements, while the stage
still describes all 45 in full.

Layout (real-world apple-orchard geometry, Z-up, metres):
  * 3 rows x 15 trees = 45 trees.
  * In-row spacing and between-row spacing are set to realistic semi-dwarf/standard values
    (defaults 4.5 m in-row, 6.0 m between rows) so a tractor row and canopy gaps look right.
  * Each placement gets a deterministic heading rotation so the identical prototypes don't
    look copy-pasted.

Each unique tree is the texture-accurate mesh from `usd_export.py` (real bark/leaf/fruit
UsdPreviewSurface materials, leaf alpha cutout), so the orchard is textured, not tinted.
"""

import argparse
import json
import os
import random
import sys
import time

from pyhelios import Context, PlantArchitecture

from usd_export import build_tree, extract_mesh, copy_textures, write_tree_usda


def build_unique_trees(out_dir, n_unique, age_days, base_seed):
    """Build + export `n_unique` distinct fully-grown trees into out_dir/trees/. Returns list
    of dicts: {file, root, num_points, num_faces, seed}."""
    trees_dir = os.path.join(out_dir, "trees")
    os.makedirs(trees_dir, exist_ok=True)
    trees = []
    for k in range(n_unique):
        seed = base_seed + k
        t0 = time.time()
        with Context() as context:
            with PlantArchitecture(context) as plantarch:
                uuids = build_tree(context, plantarch, age_days, seed)
            mesh = extract_mesh(context, uuids)
        # Textures copied once into trees/textures; every tree references ./textures/*.
        tex_srcs = copy_textures(mesh, trees_dir)
        fname = f"tree_{k:02d}.usda"
        fpath = os.path.join(trees_dir, fname)
        write_tree_usda(fpath, mesh, "textures", tex_srcs, root_prim="AppleTree")
        trees.append({
            "file": f"trees/{fname}", "root": "AppleTree",
            "num_points": len(mesh["points"]), "num_faces": len(mesh["face_counts"]),
            "seed": seed, "build_export_s": round(time.time() - t0, 2),
            "size_mb": round(os.path.getsize(fpath) / 1e6, 2),
        })
        print(f"  tree_{k:02d}: seed={seed} faces={trees[-1]['num_faces']} "
              f"size={trees[-1]['size_mb']}MB ({trees[-1]['build_export_s']}s)")
    return trees


def write_orchard(path, trees, rows, per_row, in_row_spacing, row_spacing, seed=7):
    """Write orchard.usda: instanceable references to the unique trees on a grid.

    Returns (n_trees, total_points, total_faces)."""
    rng = random.Random(seed)
    n_trees = rows * per_row
    # Centre the block on the origin so the camera framing is symmetric.
    x0 = -(per_row - 1) * in_row_spacing / 2.0
    y0 = -(rows - 1) * row_spacing / 2.0

    total_points = total_faces = 0
    with open(path, "w") as f:
        w = f.write
        w("#usda 1.0\n(\n")
        w('    defaultPrim = "Orchard"\n')
        w("    metersPerUnit = 1\n")
        w('    upAxis = "Z"\n)\n\n')
        w('def Xform "Orchard"\n{\n')
        idx = 0
        for r in range(rows):
            for c in range(per_row):
                tree = trees[idx % len(trees)]
                x = x0 + c * in_row_spacing
                y = y0 + r * row_spacing
                heading = rng.uniform(0.0, 360.0)  # deterministic per seed
                name = f"Tree_r{r:d}_c{c:02d}"
                w(f'    def "{name}" (\n')
                w(f'        prepend references = @./{tree["file"]}@\n')
                w("        instanceable = true\n")
                w("    )\n    {\n")
                w(f"        double3 xformOp:translate = ({x:.4f}, {y:.4f}, 0)\n")
                w(f"        float xformOp:rotateZ = {heading:.2f}\n")
                w('        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ"]\n')
                w("    }\n")
                total_points += tree["num_points"]
                total_faces += tree["num_faces"]
                idx += 1
        w("}\n")
    return n_trees, total_points, total_faces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--per-row", type=int, default=15)
    ap.add_argument("--in-row-spacing", type=float, default=4.5)  # m, semi-dwarf apple
    ap.add_argument("--row-spacing", type=float, default=6.0)     # m, tractor alley
    ap.add_argument("--age-days", type=float, default=1825.0)     # fully grown (~4.4 m)
    ap.add_argument("--n-unique", type=int, default=3)            # distinct prototypes
    ap.add_argument("--base-seed", type=int, default=9001)
    ap.add_argument(
        "--out", default=os.path.join(os.path.dirname(__file__), "output", "orchard.usda")
    )
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Building {args.n_unique} unique fully-grown trees (age {args.age_days}d)...")
    t0 = time.time()
    trees = build_unique_trees(out_dir, args.n_unique, args.age_days, args.base_seed)
    build_s = time.time() - t0

    n_trees, total_points, total_faces = write_orchard(
        args.out, trees, args.rows, args.per_row,
        args.in_row_spacing, args.row_spacing,
    )
    size_mb = os.path.getsize(args.out) / 1e6
    unique_mb = sum(t["size_mb"] for t in trees)

    row_len = (args.per_row - 1) * args.in_row_spacing
    block_w = (args.rows - 1) * args.row_spacing

    report = {
        "out": args.out, "orchard_size_mb": round(size_mb, 4),
        "unique_trees_total_mb": round(unique_mb, 2),
        "n_instances": n_trees, "rows": args.rows, "per_row": args.per_row,
        "in_row_spacing_m": args.in_row_spacing, "row_spacing_m": args.row_spacing,
        "row_length_m": round(row_len, 2), "block_width_m": round(block_w, 2),
        "age_days": args.age_days, "growth_phase": "fully_grown",
        "n_unique_prototypes": args.n_unique,
        "per_tree_num_points": trees[0]["num_points"],
        "per_tree_num_faces": trees[0]["num_faces"],
        "total_num_points": total_points, "total_num_faces": total_faces,
        "unique_trees": trees,
        "timing_s": {"build_export_all_trees": round(build_s, 2)},
    }
    report_path = os.path.splitext(args.out)[0] + "_export_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nWrote orchard: {args.out}  ({size_mb:.3f} MB, instanced)")
    print(f"Unique tree geometry on disk: {unique_mb:.1f} MB across {args.n_unique} trees")
    print(f"Orchard block: {row_len:.1f} m rows x {block_w:.1f} m across, {n_trees} trees")


if __name__ == "__main__":
    sys.exit(main())
