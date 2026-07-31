"""
Real camera-pose accuracy for MapAnything's predictions vs Phase 7's real
ground-truth Helios camera poses, for both scenes and both conditioning
levels actually run.

Method mirrors Phase 7's T7.2 (D1) conditioning-axis logic
(`yogesh_dev/phase7/mv_geometry.py`, reused directly via sys.path
injection, not re-derived):
  - "full" conditioning (real intrinsics+extrinsics+depth given): the
    predicted world frame IS the real Helios world frame (poses were given
    as strong conditioning, not estimated) -- direct RMSE, no alignment,
    same reasoning as T7.2 conditions C/D (`rmse_rigid`).
  - "images_only" conditioning: MapAnything estimates its own poses in an
    arbitrary frame -- Umeyama similarity alignment (rotation + isotropic
    scale + translation) of predicted camera centers onto real camera
    centers, exactly as T7.2 condition B used for its own arbitrary-frame
    reconstruction. The recovered scale factor is reported explicitly
    (not hidden), same as T7.2's `recovered_scale_factor` finding.
"""

import json
import os
import sys

import numpy as np

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE7_DIR = os.path.join(REPO_DIR, "..", "phase7")
sys.path.insert(0, os.path.abspath(PHASE7_DIR))
from mv_geometry import umeyama_alignment, rmse_rigid  # noqa: E402

OUTPUT_DIR = os.path.join(REPO_DIR, "output")


def _load_run(scene_name, conditioning):
    run_dir = os.path.join(OUTPUT_DIR, scene_name, conditioning)
    with open(os.path.join(run_dir, "run_meta.json")) as f:
        meta = json.load(f)
    npz = np.load(os.path.join(run_dir, "predictions.npz"))
    n_views = meta["n_views"]
    pred_camera_poses = [npz[f"camera_poses__{i}"] for i in range(n_views)]
    return meta, pred_camera_poses


def _rot_angle_deg(R_a, R_b):
    """Angular difference (deg) between two rotation matrices."""
    R_rel = R_a.T @ R_b
    cos_theta = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def compare_scene_conditioning(scene_name, conditioning):
    meta, pred_poses = _load_run(scene_name, conditioning)
    gt_poses_raw = [np.array(p) for p in meta["gt_pose_c2w"]]
    n = len(gt_poses_raw)

    # MapAnything anchors its output world frame to view 0's OWN camera frame
    # (verified empirically: predicted camera_poses[0] is ~identity even when
    # real, non-identity absolute extrinsics were given as "full" conditioning
    # -- the model uses given camera_poses to fix RELATIVE geometry between
    # views, not to preserve the caller's absolute world frame). Re-reference
    # both predicted and ground-truth poses to camera 0 before comparing, so
    # both sides describe the same frame (camera 0's OpenCV camera frame).
    gt_poses = [np.linalg.inv(gt_poses_raw[0]) @ p for p in gt_poses_raw]
    pred_poses = [np.linalg.inv(pred_poses[0]) @ p for p in pred_poses]

    pred_centers = np.array([p[:3, 3] for p in pred_poses])
    gt_centers = np.array([p[:3, 3] for p in gt_poses])

    if conditioning == "full":
        # Real world frame given as conditioning -- direct comparison, no alignment.
        center_rmse_m = rmse_rigid(pred_centers, gt_centers)
        rot_errs_deg = [
            _rot_angle_deg(pred_poses[i][:3, :3], gt_poses[i][:3, :3]) for i in range(n)
        ]
        result = {
            "scene_name": scene_name,
            "conditioning": conditioning,
            "method": "direct RMSE (real poses given as conditioning, same world frame expected -- "
            "T7.2 condition C/D reasoning)",
            "n_views": n,
            "camera_center_rmse_m": center_rmse_m,
            "camera_center_rmse_mm": center_rmse_m * 1000.0,
            "rotation_error_deg_mean": float(np.mean(rot_errs_deg)),
            "rotation_error_deg_max": float(np.max(rot_errs_deg)),
        }
    else:
        align = umeyama_alignment(pred_centers, gt_centers, with_scale=True)
        result = {
            "scene_name": scene_name,
            "conditioning": conditioning,
            "method": "Umeyama similarity alignment of predicted camera centers onto real "
            "camera centers, then RMSE of aligned centers -- T7.2 condition B reasoning "
            "(own-frame reconstruction, absolute scale not directly comparable without "
            "alignment; recovered_scale_factor reported explicitly, not hidden)",
            "n_views": n,
            "recovered_scale_factor": align["s"],
            "camera_center_rmse_after_alignment_m": align["rmse"],
            "camera_center_rmse_after_alignment_mm": align["rmse"] * 1000.0,
        }
    return result


def main():
    results = {}
    for scene_name in ["tree0_orbit16", "tree_t76_mvs_rig"]:
        for conditioning in ["images_only", "full"]:
            key = f"{scene_name}__{conditioning}"
            try:
                results[key] = compare_scene_conditioning(scene_name, conditioning)
                print(key, "->", json.dumps(results[key], indent=2))
            except FileNotFoundError as e:
                print(f"SKIP {key}: {e}")

    out_path = os.path.join(OUTPUT_DIR, "pose_accuracy_report.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
