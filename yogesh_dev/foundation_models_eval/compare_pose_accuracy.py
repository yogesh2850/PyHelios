"""
Real camera-pose accuracy for any model that wrote a shared prediction bundle
(DA3, Pi3X, VGGT), vs Phase 7's real ground-truth Helios camera poses.

Same Phase 7 T7.2 reasoning and the same `mv_geometry.umeyama_alignment` /
`rmse_rigid` imported directly from `yogesh_dev/phase7` that
`yogesh_dev/mapanything_eval/compare_pose_accuracy.py` used, so numbers here sit
on the same axis as MapAnything's already-published ones.

Both poses are re-referenced to camera 0 before anything is compared (the
lesson the MapAnything eval learned the hard way -- MapAnything anchors its
output world frame to view 0's own camera frame). BOTH residuals are then
always reported:

  - `camera_center_rmse_direct_mm`: no alignment at all. For a model that
    honours the caller's gauge this is the real error; for one that does not,
    this number IS the finding.
  - `camera_center_rmse_after_similarity_alignment_mm` + the recovered scale
    factor: shape accuracy with the gauge freedom removed, T7.2 condition-B
    style, with the gauge itself reported rather than hidden.

Which one is the headline number is decided empirically per run by
`frame_policy.analyse` -- see that module for why hardcoding it per model would
have been wrong.
"""

import argparse
import json
import os
import sys

import numpy as np

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE7_DIR = os.path.abspath(os.path.join(REPO_DIR, "..", "phase7"))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, PHASE7_DIR)
from mv_geometry import umeyama_alignment  # noqa: E402

import frame_policy  # noqa: E402
from prediction_bundle import load_bundle, output_root  # noqa: E402

# Conditioning levels at which a model was actually GIVEN real extrinsics.
POSE_CONDITIONED = {"full", "pose_intrinsics"}


def _rot_angle_deg(R_a, R_b):
    R_rel = R_a.T @ R_b
    return float(np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))))


def compare(model, scene, conditioning):
    meta, per_view = load_bundle(REPO_DIR, model, scene, conditioning)
    gt_raw = [np.array(p, dtype=np.float64) for p in meta["gt_pose_c2w"]]
    pred_raw = [np.array(pv["camera_poses"], dtype=np.float64) for pv in per_view]
    n = len(gt_raw)

    pol = frame_policy.analyse(gt_raw, pred_raw, umeyama_alignment)
    gt_ref, pred_ref = pol["gt_poses_ref"], pol["pred_poses_ref"]

    rot_direct = [_rot_angle_deg(pred_ref[i][:3, :3], gt_ref[i][:3, :3]) for i in range(n)]
    rot_aligned = [
        _rot_angle_deg(pol["R"] @ pred_ref[i][:3, :3], gt_ref[i][:3, :3]) for i in range(n)
    ]

    return {
        "model": model,
        "scene_name": scene,
        "conditioning": conditioning,
        "was_given_real_extrinsics": conditioning in POSE_CONDITIONED,
        "n_views": n,
        "infer_s": meta.get("infer_s"),
        "infer_s_per_view": meta.get("infer_s_per_view"),
        "model_load_s": meta.get("model_load_s"),
        "frame_policy": pol["policy"],
        "recovered_scale_factor": pol["recovered_scale_factor"],
        "camera_center_rmse_direct_mm": pol["camera_center_rmse_direct_mm"],
        "camera_center_rmse_after_similarity_alignment_mm": pol[
            "camera_center_rmse_after_similarity_alignment_mm"
        ],
        "headline_camera_center_rmse_mm": (
            pol["camera_center_rmse_direct_mm"]
            if pol["preserved"]
            else pol["camera_center_rmse_after_similarity_alignment_mm"]
        ),
        "rotation_error_deg_mean_direct": float(np.mean(rot_direct)),
        "rotation_error_deg_max_direct": float(np.max(rot_direct)),
        "rotation_error_deg_mean_after_alignment": float(np.mean(rot_aligned)),
        "rotation_error_deg_max_after_alignment": float(np.max(rot_aligned)),
        "mask_valid_fraction": meta.get("mask_valid_fraction"),
    }


def discover(model):
    """Every (scene, conditioning) bundle this model actually wrote."""
    root = os.path.join(output_root(REPO_DIR), model)
    found = []
    if not os.path.isdir(root):
        return found
    for scene in sorted(os.listdir(root)):
        sdir = os.path.join(root, scene)
        if not os.path.isdir(sdir):
            continue
        for cond in sorted(os.listdir(sdir)):
            if os.path.exists(os.path.join(sdir, cond, "run_meta.json")):
                found.append((scene, cond))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()

    results = {}
    for model in args.models:
        for scene, cond in discover(model):
            key = f"{model}__{scene}__{cond}"
            r = compare(model, scene, cond)
            results[key] = r
            print(
                f"{key}: policy={r['frame_policy']} "
                f"direct={r['camera_center_rmse_direct_mm']:.1f}mm "
                f"aligned={r['camera_center_rmse_after_similarity_alignment_mm']:.1f}mm "
                f"scale={r['recovered_scale_factor']:.3f} "
                f"rot_aligned={r['rotation_error_deg_mean_after_alignment']:.2f}deg"
            )

    out_path = os.path.join(output_root(REPO_DIR), "pose_accuracy_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing.update(results)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
