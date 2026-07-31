"""
Export MapAnything's real predicted point clouds (masked, merged across all
views, transformed into the real Helios world frame using the same rigid
transform derived in compare_voxel_reconstruction.py) as plain .npy and .ply
files -- the actual "what MapAnything predicts" artifact, viewable in any
point-cloud tool (MeshLab, CloudCompare, open3d).
"""

import os
import sys

import numpy as np

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)
from compare_voxel_reconstruction import build_world_points, PHASE7_OUTPUT_DIR  # noqa: E402

OUTPUT_DIR = os.path.join(REPO_DIR, "output")


def write_ply(path, points):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def main():
    for scene_name in ["tree0_orbit16", "tree_t76_mvs_rig"]:
        for conditioning in ["images_only", "full"]:
            run_dir = os.path.join(OUTPUT_DIR, scene_name, conditioning)
            if not os.path.isdir(run_dir):
                continue
            # build_world_points is scene-agnostic via SCENE global in that
            # module for the ground-truth-comparison path (tree_t76_mvs_rig
            # only, since that's the one with saved GT camera_poses used to
            # anchor the transform); for tree0_orbit16 we still have its own
            # real ground-truth camera_poses in run_meta.json, so reuse the
            # same logic generically here instead of importing the
            # scene-locked helper.
            import json

            with open(os.path.join(run_dir, "run_meta.json")) as f:
                meta = json.load(f)
            npz = np.load(os.path.join(run_dir, "predictions.npz"))
            n = meta["n_views"]
            gt_poses_raw = [np.array(p) for p in meta["gt_pose_c2w"]]
            pred_poses_raw = [npz[f"camera_poses__{i}"] for i in range(n)]
            combined = gt_poses_raw[0] @ np.linalg.inv(pred_poses_raw[0])

            all_pts = []
            for i in range(n):
                mask = npz[f"mask__{i}"]
                mask = mask.squeeze(-1) if mask.ndim == 3 else mask
                pts = npz[f"pts3d__{i}"][mask]
                if len(pts) == 0:
                    continue
                pts_world = (combined[:3, :3] @ pts.T).T + combined[:3, 3]
                all_pts.append(pts_world)
            merged = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3))

            np.save(os.path.join(run_dir, "world_points_merged.npy"), merged)
            # Subsample for the .ply so files stay reasonably sized for viewers.
            if len(merged) > 500_000:
                idx = np.random.default_rng(0).choice(len(merged), 500_000, replace=False)
                ply_pts = merged[idx]
            else:
                ply_pts = merged
            write_ply(os.path.join(run_dir, "world_points_merged.ply"), ply_pts)
            print(f"{scene_name}/{conditioning}: {len(merged)} points -> world_points_merged.npy/.ply "
                  f"(note: 'full' conditioning world frame is real-metric via view0 anchor; "
                  f"'images_only' is NOT rescaled here -- see compare_voxel_reconstruction.py "
                  f"for the Umeyama-aligned version used in the accuracy reports)")


if __name__ == "__main__":
    main()
