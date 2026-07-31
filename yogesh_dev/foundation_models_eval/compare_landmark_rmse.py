"""
Real 3D landmark-recovery RMSE for any model that wrote a shared prediction
bundle, on `tree_t76_mvs_rig`'s real 42-view WAI scene.

Logic is a direct port of `yogesh_dev/mapanything_eval/compare_landmark_rmse.py`
(same real landmarks -- T7.6's branch tube-segment midpoints + real fruit
centroids from `t76_branch_segments.json` / `t76_ground_truth_scale.json`;
same camera-0 re-referencing; same projection-and-sample method; same Umeyama
handling for unconditioned levels; same Phase 7 T7.2 comparison block), so the
mm numbers it prints sit on the same axis as MapAnything's already-published
13872.0 mm (images_only) / 398.8 mm (full) and Phase 7's classical proxy
74.4 mm (condition A) / 4.6 mm (condition C).

For each real landmark, in each real view: project it using the MODEL's OWN
reported per-view intrinsics/pose (not the original image K, since every model
here works on its own internally-resized grid), sample that view's predicted
world point at the nearest output pixel, gated on the MODEL's OWN validity mask
-- i.e. the occlusion/validity signal actually available at inference time, never
ground-truth depth.
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
from compare_pose_accuracy import POSE_CONDITIONED, discover  # noqa: E402
from prediction_bundle import load_bundle, output_root  # noqa: E402

PHASE7_OUTPUT_DIR = os.path.join(PHASE7_DIR, "output")
SCENE = "tree_t76_mvs_rig"


def load_landmarks():
    with open(os.path.join(PHASE7_OUTPUT_DIR, "t76_branch_segments.json")) as f:
        segments = json.load(f)
    with open(os.path.join(PHASE7_OUTPUT_DIR, "t76_ground_truth_scale.json")) as f:
        gt_scale = json.load(f)
    branch = [
        {"point": ((np.array(s["p0"]) + np.array(s["p1"])) / 2.0).tolist(), "kind": "branch"}
        for s in segments
    ]
    fruit = [{"point": fr["point"], "kind": "fruit"} for fr in gt_scale["fruit"]]
    return branch + fruit


def _project(K, pose_c2w, point_world):
    pose_w2c = np.linalg.inv(pose_c2w)
    p_cam = pose_w2c @ np.array([*point_world, 1.0])
    z = p_cam[2]
    if z <= 1e-6:
        return None
    uvw = K @ (p_cam[:3] / z)
    return uvw[0], uvw[1], z


def landmark_rmse(model, conditioning, landmarks):
    meta, per_view = load_bundle(REPO_DIR, model, SCENE, conditioning)
    gt_poses_raw = [np.array(p, dtype=np.float64) for p in meta["gt_pose_c2w"]]
    n = meta["n_views"]
    H, W = per_view[0]["pts3d"].shape[0], per_view[0]["pts3d"].shape[1]

    T0_gt_inv = np.linalg.inv(gt_poses_raw[0])
    pred_raw = [np.array(pv["camera_poses"], dtype=np.float64) for pv in per_view]
    T0_pred_inv = np.linalg.inv(pred_raw[0])

    # Whether the model preserved the caller's absolute gauge is decided
    # empirically, not assumed from the conditioning level -- DA3 preserves it
    # exactly, Pi3X does not even when handed real poses AND real metric depth.
    pol = frame_policy.analyse(gt_poses_raw, pred_raw, umeyama_alignment)
    gt_poses_ref, pred_poses_ref = pol["gt_poses_ref"], pol["pred_poses_ref"]

    def to_gt_frame(p):
        if pol["preserved"]:
            return p
        return pol["s"] * (pol["R"] @ p) + pol["t"]

    landmark_world = np.array([lm["point"] for lm in landmarks])
    landmark_ref = (T0_gt_inv[:3, :3] @ landmark_world.T).T + T0_gt_inv[:3, 3]

    errors = []
    kinds = []
    n_valid_obs = 0
    n_with_obs = 0
    for lm_idx, L_ref in enumerate(landmark_ref):
        # Predicted points live in the model's own native frame; re-reference
        # them to camera 0 the same way the poses were.
        got_one = False
        for v in range(n):
            proj = _project(per_view[v]["intrinsics"], pred_poses_ref[v], L_ref)
            if proj is None:
                continue
            u, vc, _z = proj
            ui, vi = int(round(u)), int(round(vc))
            if not (0 <= ui < W and 0 <= vi < H):
                continue
            if not per_view[v]["mask"][vi, ui]:
                continue
            p_native = np.array(per_view[v]["pts3d"][vi, ui], dtype=np.float64)
            p_ref = T0_pred_inv[:3, :3] @ p_native + T0_pred_inv[:3, 3]
            err = float(np.linalg.norm(to_gt_frame(p_ref) - L_ref))
            errors.append(err)
            kinds.append(landmarks[lm_idx]["kind"])
            n_valid_obs += 1
            got_one = True
        if got_one:
            n_with_obs += 1

    errors = np.array(errors)
    kinds = np.array(kinds)
    rmse_m = float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None

    def _sub(kind):
        e = errors[kinds == kind] if len(errors) else np.array([])
        return float(np.sqrt(np.mean(e ** 2))) * 1000.0 if len(e) else None

    return {
        "model": model,
        "scene_name": SCENE,
        "conditioning": conditioning,
        "n_landmarks_total": len(landmarks),
        "n_landmarks_with_any_valid_observation": n_with_obs,
        "n_valid_landmark_view_observations": n_valid_obs,
        "rmse_mm": rmse_m * 1000.0 if rmse_m is not None else None,
        "median_error_mm": float(np.median(errors)) * 1000.0 if len(errors) else None,
        "branch_rmse_mm": _sub("branch"),
        "fruit_rmse_mm": _sub("fruit"),
        "frame_policy": pol["policy"],
        "recovered_scale_factor": None if pol["preserved"] else pol["recovered_scale_factor"],
        "was_given_real_extrinsics": conditioning in POSE_CONDITIONED,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    landmarks = load_landmarks()

    with open(os.path.join(PHASE7_OUTPUT_DIR, "t72_pose_conditioning_ablation.json")) as f:
        t72 = json.load(f)

    results = {}
    for model in args.models:
        for scene, cond in discover(model):
            if scene != SCENE:
                continue
            key = f"{model}__{cond}"
            results[key] = landmark_rmse(model, cond, landmarks)
            r = results[key]
            print(
                f"{key}: rmse={r['rmse_mm']}, median={r['median_error_mm']}, "
                f"n_obs={r['n_valid_landmark_view_observations']}"
            )

    out_path = os.path.join(output_root(REPO_DIR), "landmark_rmse_report.json")
    payload = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            payload = json.load(f)
    payload.setdefault("per_model_conditioning", {}).update(results)
    payload["reference_numbers"] = {
        "phase7_classical_proxy_t72_A_images_only_rmse_mm": t72["condition_A_images_only"][
            "rmse_after_affine_align_mm"
        ],
        "phase7_classical_proxy_t72_C_intrinsics_extrinsics_rmse_mm": t72[
            "condition_C_intrinsics_extrinsics"
        ]["rmse_mm"],
        "mapanything_images_only_rmse_mm": 13872.0,
        "mapanything_full_rmse_mm": 398.8,
        "mapanything_source": (
            "yogesh_dev/mapanything_eval/output/landmark_rmse_report.json, produced by the "
            "same method on the same scene/landmarks -- quoted here so the running "
            "comparison table has one place to read from."
        ),
        "caveat": (
            "Phase 7's T7.2 proxy used its own 8-view/180deg-arc rig and a different real "
            "tree instance, and is sparse classical triangulation of exactly the queried "
            "landmarks (best case for classical geometry); every foundation model here is "
            "dense feed-forward prediction sampled at a landmark's projected pixel. Same "
            "conditioning axis and same metric, NOT a literal same-scene/same-paradigm rig."
        ),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
