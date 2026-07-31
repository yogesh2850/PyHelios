"""
Real head-to-head against Phase 7's T7.6 (classical depth-fusion baseline)
and T7.7 (metric-scale integrity), on `tree_t76_mvs_rig` -- the exact same
real tree, real ground truth, real voxel grid definition (bmin/voxel_size,
loaded from T7.6's own persisted `t76_grid_coarse.npz`/`t76_grid_fine.npz`),
and (for branch-recall) the exact same recall functions
(`yogesh_dev/phase7/thin_structure_recall.py`, imported and called directly,
not re-derived), just with MapAnything's own predicted points fused into the
grid instead of T7.6's classical log-odds depth fusion.

## Getting MapAnything's points into the real Helios world frame

MapAnything anchors its output world frame to view 0's own camera frame
(see `compare_pose_accuracy.py`). The rigid transform that brings a point
in that native frame into the real raw Helios world frame (the frame
T7.6's grid bmin/bmax are defined in) is:

    combined = gt_pose_c2w[0] @ inv(pred_camera_poses[0])

For "full" conditioning this is verified consistent: applying `combined` to
MapAnything's OTHER predicted camera poses reproduces the real ground-truth
poses to within the ~70-90mm / <1.2deg found in `compare_pose_accuracy.py`
-- i.e. `combined` is not a free parameter fit to make this comparison look
good, it is fixed entirely by view 0's real pose and MapAnything's own view
0 output.

For "images_only" conditioning, scale is not metric-correct without depth,
so `combined` is composed with the same Umeyama (rotation+scale+translation)
fit already used in `compare_landmark_rmse.py`/`compare_pose_accuracy.py`.

## Differences from T7.6's classical fusion (stated explicitly, not hidden)

T7.6's occupancy grid comes from real log-odds probabilistic depth fusion
(free-space carving + occupancy accumulation across views,
`phase4/occupancy_map.py`). This script does plain point-cloud voxelization
(any MapAnything-predicted, mask-valid point marks its voxel occupied, no
free-space carving, no multi-view consistency accumulation) -- a simpler,
weaker fusion method. A worse recall number here is not automatically "the
foundation model is worse at reconstruction" -- part of it is this
simpler fusion step. This is flagged, not blurred, exactly in the spirit of
Phase 7's own D_vs_C caveat.
"""

import json
import os
import sys

import numpy as np

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE7_DIR = os.path.abspath(os.path.join(REPO_DIR, "..", "phase7"))
sys.path.insert(0, PHASE7_DIR)
from mv_geometry import umeyama_alignment  # noqa: E402
from thin_structure_recall import _theoretical_recall, _empirical_recall  # noqa: E402

OUTPUT_DIR = os.path.join(REPO_DIR, "output")
PHASE7_OUTPUT_DIR = os.path.join(PHASE7_DIR, "output")
SCENE = "tree_t76_mvs_rig"

# Same constants as yogesh_dev/phase7/metric_scale_integrity.py (copied, not
# imported, since that module's `run()` hardcodes its own I/O paths rather
# than exposing parameterized functions -- values kept byte-identical).
FRUIT_SEARCH_RADIUS_FACTOR = 1.3
MIN_VOXELS_FOR_FRUIT_EXTENT = 4
INTERNODE_SNAP_TOL_VOXELS = 2
MAX_INTERNODE_SEGMENTS_CHECKED = 200


def _load_run(conditioning):
    run_dir = os.path.join(OUTPUT_DIR, SCENE, conditioning)
    with open(os.path.join(run_dir, "run_meta.json")) as f:
        meta = json.load(f)
    npz = np.load(os.path.join(run_dir, "predictions.npz"))
    n = meta["n_views"]
    per_view = [
        {"pts3d": npz[f"pts3d__{i}"], "mask": npz[f"mask__{i}"], "camera_poses": npz[f"camera_poses__{i}"]}
        for i in range(n)
    ]
    gt_poses_raw = [np.array(p) for p in meta["gt_pose_c2w"]]
    return meta, per_view, gt_poses_raw


def _rigid_transform_points(T, pts):
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def build_world_points(conditioning):
    meta, per_view, gt_poses_raw = _load_run(conditioning)
    pred_poses_raw = [pv["camera_poses"] for pv in per_view]

    combined = gt_poses_raw[0] @ np.linalg.inv(pred_poses_raw[0])

    scale = 1.0
    if conditioning == "images_only":
        T0_pred_inv = np.linalg.inv(pred_poses_raw[0])
        T0_gt_inv = np.linalg.inv(gt_poses_raw[0])
        pred_centers_ref = np.array([(T0_pred_inv @ p)[:3, 3] for p in pred_poses_raw])
        gt_centers_ref = np.array([(T0_gt_inv @ p)[:3, 3] for p in gt_poses_raw])
        align = umeyama_alignment(pred_centers_ref, gt_centers_ref, with_scale=True)
        # combined_with_scale: native -> camera0 (T0_pred_inv) -> similarity align (R,s,t
        # onto GT-camera0-frame) -> real world (gt_poses_raw[0])
        scale = align["s"]

        def transform(pts_native):
            p_cam0 = _rigid_transform_points(T0_pred_inv, pts_native)
            p_gt_cam0 = scale * (align["R"] @ p_cam0.T).T + align["t"]
            return _rigid_transform_points(gt_poses_raw[0], p_gt_cam0)
    else:

        def transform(pts_native):
            return _rigid_transform_points(combined, pts_native)

    all_points = []
    for pv in per_view:
        mask = pv["mask"].squeeze(-1) if pv["mask"].ndim == 3 else pv["mask"]
        pts = pv["pts3d"][mask]
        if len(pts) == 0:
            continue
        all_points.append(transform(pts))
    world_points = np.concatenate(all_points, axis=0) if all_points else np.zeros((0, 3))
    return world_points, scale


def voxelize(points, bmin, voxel_size, dims):
    if len(points) == 0:
        return np.zeros(tuple(dims), dtype=bool)
    idx = np.floor((points - bmin[None, :]) / voxel_size).astype(np.int64)
    valid = np.all((idx >= 0) & (idx < dims[None, :]), axis=1)
    idx = idx[valid]
    occupied = np.zeros(tuple(dims), dtype=bool)
    occupied[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occupied


def fruit_and_internode_check(occupied, bmin, voxel_m, gt, segments):
    dims = np.array(occupied.shape)
    occ_idx = np.argwhere(occupied)
    occ_world = bmin[None, :] + (occ_idx + 0.5) * voxel_m

    recon_canopy_volume_m3 = len(occ_idx) * (voxel_m ** 3)
    canopy_rel_err = (
        (recon_canopy_volume_m3 - gt["canopy_bbox_volume_m3"]) / gt["canopy_bbox_volume_m3"]
        if gt["canopy_bbox_volume_m3"] else None
    )

    fruit_results = []
    for fr in gt["fruit"]:
        center = np.array(fr["point"])
        gt_diam = fr["diameter_m"]
        search_r = FRUIT_SEARCH_RADIUS_FACTOR * (gt_diam / 2.0)
        d = np.linalg.norm(occ_world - center[None, :], axis=1) if len(occ_world) else np.array([])
        nearby_d = d[d <= search_r]
        if len(nearby_d) < MIN_VOXELS_FOR_FRUIT_EXTENT:
            fruit_results.append({"gt_diameter_m": gt_diam, "recon_diameter_m": None,
                                   "n_nearby_voxels": int(len(nearby_d)), "skipped": True})
            continue
        recon_diam = float(2.0 * nearby_d.max())
        fruit_results.append({
            "gt_diameter_m": gt_diam, "recon_diameter_m": recon_diam,
            "n_nearby_voxels": int(len(nearby_d)), "skipped": False,
            "rel_err": (recon_diam - gt_diam) / gt_diam,
        })
    used_fruit = [r for r in fruit_results if not r["skipped"]]
    mean_fruit_rel_err = float(np.mean([abs(r["rel_err"]) for r in used_fruit])) if used_fruit else None

    rng = np.random.default_rng(0)
    shoot_segs = [s for s in segments if s["label"] == "shoot"]
    if len(shoot_segs) > MAX_INTERNODE_SEGMENTS_CHECKED:
        idx = rng.choice(len(shoot_segs), MAX_INTERNODE_SEGMENTS_CHECKED, replace=False)
        shoot_segs = [shoot_segs[i] for i in idx]

    def snap_to_occupied(point):
        idx = np.floor((point - bmin) / voxel_m).astype(np.int64)
        lo = np.clip(idx - INTERNODE_SNAP_TOL_VOXELS, 0, dims - 1)
        hi = np.clip(idx + INTERNODE_SNAP_TOL_VOXELS + 1, 0, dims)
        if np.any(lo >= hi):
            return None
        window = occupied[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        local_idx = np.argwhere(window)
        if len(local_idx) == 0:
            return None
        world_candidates = bmin[None, :] + (lo[None, :] + local_idx + 0.5) * voxel_m
        d = np.linalg.norm(world_candidates - point[None, :], axis=1)
        return world_candidates[np.argmin(d)]

    internode_lengths_recon = []
    n_checked, n_snapped = 0, 0
    for seg in shoot_segs:
        n_checked += 1
        p0, p1 = np.array(seg["p0"]), np.array(seg["p1"])
        s0, s1 = snap_to_occupied(p0), snap_to_occupied(p1)
        if s0 is None or s1 is None:
            continue
        n_snapped += 1
        internode_lengths_recon.append(float(np.linalg.norm(s1 - s0)))

    recon_mean_internode = float(np.mean(internode_lengths_recon)) if internode_lengths_recon else None
    internode_rel_err = (
        (recon_mean_internode - gt["mean_internode_length_m"]) / gt["mean_internode_length_m"]
        if recon_mean_internode is not None else None
    )

    return {
        "canopy_volume": {
            "ground_truth_bbox_m3": gt["canopy_bbox_volume_m3"],
            "reconstructed_occupied_voxel_volume_m3": recon_canopy_volume_m3,
            "relative_error": canopy_rel_err,
        },
        "fruit_diameter": {
            "n_fruit_total": len(gt["fruit"]), "n_fruit_used": len(used_fruit),
            "mean_gt_diameter_m": gt["mean_fruit_diameter_m"],
            "mean_absolute_relative_error": mean_fruit_rel_err,
        },
        "internode_length": {
            "n_segments_checked": n_checked, "n_segments_snapped_both_ends": n_snapped,
            "ground_truth_mean_m": gt["mean_internode_length_m"],
            "reconstructed_mean_m": recon_mean_internode,
            "relative_error": internode_rel_err,
        },
    }


def run_for_conditioning(conditioning, segments, gt_scale):
    world_points, scale = build_world_points(conditioning)

    results_by_res = {}
    for res_name, npz_name in [("coarse_2cm", "t76_grid_coarse.npz"), ("fine_5mm", "t76_grid_fine.npz")]:
        grid = np.load(os.path.join(PHASE7_OUTPUT_DIR, npz_name))
        bmin, voxel_size, dims = grid["bmin"], float(grid["voxel_size"]), np.array(grid["occupied"].shape)
        classical_occupied = grid["occupied"]

        mapanything_occupied = voxelize(world_points, bmin, voxel_size, dims)

        theo_recall, theo_recall_robust, by_class_total = _theoretical_recall(segments, voxel_size)
        emp_recall, _t2, by_class_hit = _empirical_recall(segments, mapanything_occupied, bmin, voxel_size)
        classical_emp_recall, _t3, classical_by_class_hit = _empirical_recall(
            segments, classical_occupied, bmin, voxel_size
        )

        results_by_res[res_name] = {
            "voxel_size_m": voxel_size,
            "n_occupied_voxels_mapanything": int(mapanything_occupied.sum()),
            "n_occupied_voxels_classical_t76": int(classical_occupied.sum()),
            "n_points_fused": int(len(world_points)),
            "theoretical_geometric_recall_one_voxel": theo_recall,
            "empirical_reconstruction_recall_mapanything": emp_recall,
            "empirical_reconstruction_recall_classical_t76": classical_emp_recall,
        }

    scale_and_internode = fruit_and_internode_check(
        mapanything_occupied, bmin, voxel_size, gt_scale, segments
    )  # uses the fine_5mm grid (last iteration above)

    return {
        "conditioning": conditioning,
        "recovered_scale_factor_applied": scale,
        "by_resolution": results_by_res,
        "metric_scale_integrity": scale_and_internode,
    }


def main():
    with open(os.path.join(PHASE7_OUTPUT_DIR, "t76_branch_segments.json")) as f:
        segments = json.load(f)
    with open(os.path.join(PHASE7_OUTPUT_DIR, "t76_ground_truth_scale.json")) as f:
        gt_scale = json.load(f)
    with open(os.path.join(PHASE7_OUTPUT_DIR, "t76_classical_baseline.json")) as f:
        t76_classical = json.load(f)
    with open(os.path.join(PHASE7_OUTPUT_DIR, "t77_metric_scale_integrity.json")) as f:
        t77_classical = json.load(f)

    report = {}
    for conditioning in ["full", "images_only"]:
        report[conditioning] = run_for_conditioning(conditioning, segments, gt_scale)
        print(f"--- {conditioning} ---")
        print(json.dumps(report[conditioning]["by_resolution"]["fine_5mm"], indent=2, default=float))
        print(json.dumps(report[conditioning]["metric_scale_integrity"], indent=2, default=float))

    report["vs_phase7_t76_t77_classical_baseline"] = {
        "branch_recovery_rate_fine_grid_classical_t76": t76_classical["branch_recovery_rate_fine_grid"],
        "fruit_recovery_rate_fine_grid_classical_t76": t76_classical["fruit_recovery_rate_fine_grid"],
        "fruit_diameter_mean_abs_rel_err_classical_t77": t77_classical["fruit_diameter"]["mean_absolute_relative_error"],
        "internode_length_rel_err_classical_t77": t77_classical["internode_length"]["relative_error"],
        "caveat": (
            "T7.6's classical baseline used real log-odds probabilistic depth fusion "
            "with free-space carving (phase4/occupancy_map.py); this script uses plain "
            "point-cloud voxelization of MapAnything's mask-valid predicted points (no "
            "free-space carving/multi-view consistency accumulation) -- a simpler, "
            "weaker fusion step on the MapAnything side, stated explicitly so a lower "
            "MapAnything recall number is not misread as purely a foundation-model "
            "shortfall."
        ),
    }

    out_path = os.path.join(OUTPUT_DIR, "voxel_reconstruction_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(json.dumps(report["vs_phase7_t76_t77_classical_baseline"], indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
