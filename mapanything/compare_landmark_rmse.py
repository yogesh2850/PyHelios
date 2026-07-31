"""
Real landmark-recovery RMSE for MapAnything, directly comparable in
methodology (not literal rig/tree) to Phase 7's T7.2 (D1) pose-conditioning
ablation (`t72_pose_conditioning_ablation.json`) and T7.3 (D2) baseline-angle
sweep -- same real 3D landmarks (branch tube-segment midpoints + real fruit
centroids), same real Helios tree used by T7.6/T7.7
(`t76_branch_segments.json`, `t76_ground_truth_scale.json`), evaluated
against `tree_t76_mvs_rig`'s real 42-view WAI scene.

Method, per conditioning level actually run:
  - "full" (+intrinsics+extrinsics+depth): MapAnything's world frame is
    anchored to view 0's own camera frame (empirically verified in
    `compare_pose_accuracy.py`), which real camera pose 0 defines a known
    rigid transform of the real Helios world frame -- so after
    re-referencing ground truth to camera 0, MapAnything's output world
    frame directly corresponds to real, metric ground truth with NO further
    alignment needed. This is the T7.2 condition-C/D regime (known
    extrinsics -> real metric-scale comparison, `rmse_rigid` reasoning).
  - "images_only": no extrinsics given, so MapAnything's overall scale is
    real but arbitrary. Reuses the exact same Umeyama similarity transform
    (rotation + isotropic scale + translation) already fit in
    `compare_pose_accuracy.py` from predicted-vs-real camera centers, and
    applies it to landmark-sampled 3D points before computing RMSE -- T7.2
    condition A/B regime.

For each real landmark, in each of the 42 real views: project the landmark
into that view using MapAnything's OWN reported per-view intrinsics/pose
(not the original image-space K, since MapAnything works on an internally
resized image grid -- using its own reported K/pose sidesteps needing to
reverse-engineer that resize transform), sample its predicted `pts3d` at
the nearest output pixel, gated on MapAnything's own predicted validity
mask (`mask`, which folds in confidence + edge detection) -- i.e. the
occlusion/validity signal actually available at inference time, not
ground-truth depth (a deliberately real-world-honest choice, see LOG.md).
"""

import json
import os
import sys

import numpy as np

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE7_DIR = os.path.abspath(os.path.join(REPO_DIR, "..", "phase7"))
sys.path.insert(0, PHASE7_DIR)
from mv_geometry import umeyama_alignment  # noqa: E402

OUTPUT_DIR = os.path.join(REPO_DIR, "output")
PHASE7_OUTPUT_DIR = os.path.join(PHASE7_DIR, "output")

SCENE = "tree_t76_mvs_rig"


def _load_landmarks():
    with open(os.path.join(PHASE7_OUTPUT_DIR, "t76_branch_segments.json")) as f:
        segments = json.load(f)
    with open(os.path.join(PHASE7_OUTPUT_DIR, "t76_ground_truth_scale.json")) as f:
        gt_scale = json.load(f)
    branch_landmarks = [
        {"point": ((np.array(s["p0"]) + np.array(s["p1"])) / 2.0).tolist(), "kind": "branch", "label": s["label"]}
        for s in segments
    ]
    fruit_landmarks = [
        {"point": fr["point"], "kind": "fruit", "label": "fruit"} for fr in gt_scale["fruit"]
    ]
    return branch_landmarks + fruit_landmarks


def _load_run(conditioning):
    run_dir = os.path.join(OUTPUT_DIR, SCENE, conditioning)
    with open(os.path.join(run_dir, "run_meta.json")) as f:
        meta = json.load(f)
    npz = np.load(os.path.join(run_dir, "predictions.npz"))
    n = meta["n_views"]
    per_view = [
        {
            "pts3d": npz[f"pts3d__{i}"],
            "mask": npz[f"mask__{i}"],
            "intrinsics": npz[f"intrinsics__{i}"],
            "camera_poses": npz[f"camera_poses__{i}"],
        }
        for i in range(n)
    ]
    return meta, per_view


def _project(K, pose_c2w, point_world):
    """point_world (3,) -> (u, v, z_cam) in the camera defined by pose_c2w (cam2world)."""
    pose_w2c = np.linalg.inv(pose_c2w)
    p_h = np.array([point_world[0], point_world[1], point_world[2], 1.0])
    p_cam = pose_w2c @ p_h
    z = p_cam[2]
    if z <= 1e-6:
        return None
    uvw = K @ (p_cam[:3] / z)
    return uvw[0], uvw[1], z


def landmark_rmse_for_conditioning(conditioning, landmarks):
    meta, per_view = _load_run(conditioning)
    gt_poses_raw = [np.array(p) for p in meta["gt_pose_c2w"]]
    n = meta["n_views"]
    H, W = per_view[0]["pts3d"].shape[0], per_view[0]["pts3d"].shape[1]

    # Re-reference both GT camera poses and predicted camera poses to camera 0
    # (see compare_pose_accuracy.py docstring for why this is necessary).
    T0_gt_inv = np.linalg.inv(gt_poses_raw[0])
    gt_poses_ref = [T0_gt_inv @ p for p in gt_poses_raw]
    pred_poses_raw = [pv["camera_poses"] for pv in per_view]
    T0_pred_inv = np.linalg.inv(pred_poses_raw[0])
    pred_poses_ref = [T0_pred_inv @ p for p in pred_poses_raw]

    align = None
    if conditioning == "images_only":
        pred_centers = np.array([p[:3, 3] for p in pred_poses_ref])
        gt_centers = np.array([p[:3, 3] for p in gt_poses_ref])
        align = umeyama_alignment(pred_centers, gt_centers, with_scale=True)

    def to_gt_frame(p_pred_ref):
        if align is None:
            return p_pred_ref
        return align["s"] * (align["R"] @ p_pred_ref) + align["t"]

    landmark_world = np.array([lm["point"] for lm in landmarks])
    # Ground truth landmarks are already in the real Helios world frame ==
    # the frame gt_poses_ref[0] is identity in (camera 0's own frame), i.e.
    # no transform needed for the GT side: gt_poses_raw already IS that
    # frame's definition up to the camera-0 anchor, and landmark_world was
    # never re-referenced -- but gt_poses_ref[0] == identity means camera 0's
    # frame IS the real world frame with camera 0 at the origin looking down
    # its real optical axis, NOT the raw Helios world frame. Bring landmarks
    # into that same frame for a consistent comparison.
    landmark_ref = (T0_gt_inv[:3, :3] @ landmark_world.T).T + T0_gt_inv[:3, 3]

    errors = []
    n_valid_obs = 0
    n_landmarks_with_any_obs = 0
    for lm_idx, L_ref in enumerate(landmark_ref):
        best_err = None
        for v in range(n):
            proj = _project(per_view[v]["intrinsics"], pred_poses_ref[v], L_ref)
            if proj is None:
                continue
            u, vcoord, z = proj
            ui, vi = int(round(u)), int(round(vcoord))
            if not (0 <= ui < W and 0 <= vi < H):
                continue
            if not per_view[v]["mask"][vi, ui]:
                continue
            p_sample_pred_ref = per_view[v]["pts3d"][vi, ui]
            p_sample_world = to_gt_frame(p_sample_pred_ref)
            err = float(np.linalg.norm(p_sample_world - L_ref))
            errors.append(err)
            n_valid_obs += 1
            if best_err is None:
                best_err = err
        if best_err is not None:
            n_landmarks_with_any_obs += 1

    errors = np.array(errors)
    rmse_m = float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None
    result = {
        "scene_name": SCENE,
        "conditioning": conditioning,
        "n_landmarks_total": len(landmarks),
        "n_landmarks_with_any_valid_observation": n_landmarks_with_any_obs,
        "n_valid_landmark_view_observations": n_valid_obs,
        "rmse_m": rmse_m,
        "rmse_mm": rmse_m * 1000.0 if rmse_m is not None else None,
        "median_error_mm": float(np.median(errors)) * 1000.0 if len(errors) else None,
        "recovered_scale_factor": align["s"] if align is not None else None,
    }
    return result


def main():
    landmarks = _load_landmarks()
    results = {}
    for conditioning in ["images_only", "full"]:
        r = landmark_rmse_for_conditioning(conditioning, landmarks)
        results[conditioning] = r
        print(json.dumps(r, indent=2))

    t72_path = os.path.join(PHASE7_OUTPUT_DIR, "t72_pose_conditioning_ablation.json")
    with open(t72_path) as f:
        t72 = json.load(f)
    comparison = {
        "mapanything": {
            "images_only_rmse_mm": results["images_only"]["rmse_mm"],
            "full_conditioning_rmse_mm": results["full"]["rmse_mm"],
        },
        "phase7_classical_proxy_t72": {
            "A_images_only_rmse_mm": t72["condition_A_images_only"]["rmse_after_affine_align_mm"],
            "C_intrinsics_extrinsics_rmse_mm": t72["condition_C_intrinsics_extrinsics"]["rmse_mm"],
        },
        "caveat": (
            "Different rig (Phase 7's T7.2 used its own 8-view/180deg-arc rig with a "
            "different real tree instance and different landmark set; this run uses "
            "T7.6's 42-view rig/tree) and different reconstruction paradigm (T7.2's "
            "proxy is sparse classical multi-view triangulation of exactly these "
            "landmarks; MapAnything is dense feed-forward pointmap prediction sampled "
            "at landmark projections) -- NOT a literal same-scene/same-points "
            "comparison. It IS a real comparison on the same conditioning-level axis "
            "(images-only vs full known-extrinsics+depth) and the same accuracy metric "
            "(3D landmark RMSE in mm) against the same category of real Helios "
            "branch/fruit ground truth."
        ),
    }
    out_path = os.path.join(OUTPUT_DIR, "landmark_rmse_report.json")
    with open(out_path, "w") as f:
        json.dump({"per_conditioning": results, "vs_phase7_t72": comparison}, f, indent=2)
    print(json.dumps(comparison, indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
