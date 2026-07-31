"""
Real voxel-grid reconstruction accuracy for any model that wrote a shared
prediction bundle, head-to-head against Phase 7's T7.6 classical depth-fusion
baseline and T7.7 metric-scale integrity, on `tree_t76_mvs_rig`.

Direct port of `yogesh_dev/mapanything_eval/compare_voxel_reconstruction.py`:
same real tree, same real ground truth, same real grid definition (bmin /
voxel_size / dims loaded from T7.6's own persisted `t76_grid_coarse.npz` and
`t76_grid_fine.npz`), same recall functions imported directly from
`yogesh_dev/phase7/thin_structure_recall.py`, and the same constants copied
byte-identically from `yogesh_dev/phase7/metric_scale_integrity.py`. Only the
source of the fused points changes.

## Getting each model's points into the real Helios world frame

    combined = gt_pose_c2w[0] @ inv(pred_camera_poses[0])

For pose-conditioned levels this is fixed entirely by view 0's real pose and
the model's own view-0 output -- not a free parameter fit to flatter the
result. For unconditioned levels it is composed with the same Umeyama
similarity fit used in `compare_pose_accuracy.py` / `compare_landmark_rmse.py`,
since scale is not metric-recoverable without conditioning.

## The fusion caveat, restated because it still applies

T7.6's occupancy grid came from real log-odds probabilistic depth fusion with
free-space carving (`phase4/occupancy_map.py`). This script does plain
point-cloud voxelization of each model's mask-valid predicted points -- a
simpler, weaker fusion method that also produces a much DENSER occupied grid.
A denser grid makes "is there an occupied voxel within one voxel of this
landmark" easier to satisfy somewhat independently of reconstruction quality,
so branch-recall wins here are partly a fusion-density artifact. The
fruit-diameter (tight GT-centered search radius) and internode-length
(nearest-occupied-voxel snap) numbers are much less sensitive to this and are
the more trustworthy reconstruction-quality signals. Flagged, not blurred --
same as the MapAnything eval did.
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
from thin_structure_recall import _empirical_recall, _theoretical_recall  # noqa: E402

import frame_policy  # noqa: E402
from compare_pose_accuracy import POSE_CONDITIONED, discover  # noqa: E402
from prediction_bundle import load_bundle, output_root  # noqa: E402

PHASE7_OUTPUT_DIR = os.path.join(PHASE7_DIR, "output")
SCENE = "tree_t76_mvs_rig"

# Copied byte-identically from yogesh_dev/phase7/metric_scale_integrity.py
# (that module's run() hardcodes its own I/O paths rather than exposing
# parameterized functions, so the constants are copied, not imported).
FRUIT_SEARCH_RADIUS_FACTOR = 1.3
MIN_VOXELS_FOR_FRUIT_EXTENT = 4
INTERNODE_SNAP_TOL_VOXELS = 2
MAX_INTERNODE_SEGMENTS_CHECKED = 200


def _rigid(T, pts):
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def build_world_points(model, conditioning):
    meta, per_view = load_bundle(REPO_DIR, model, SCENE, conditioning)
    gt_poses_raw = [np.array(p, dtype=np.float64) for p in meta["gt_pose_c2w"]]
    pred_raw = [np.array(pv["camera_poses"], dtype=np.float64) for pv in per_view]

    # Gauge handling is decided empirically per run (see frame_policy.py), not
    # assumed from the conditioning level: DA3 preserves the caller's frame
    # exactly under pose conditioning, Pi3X does not even with real poses AND
    # real metric depth supplied.
    pol = frame_policy.analyse(gt_poses_raw, pred_raw, umeyama_alignment)
    T0_pred_inv = np.linalg.inv(pred_raw[0])
    scale = 1.0 if pol["preserved"] else pol["s"]

    if pol["preserved"]:
        combined = gt_poses_raw[0] @ T0_pred_inv

        def transform(pts):
            return _rigid(combined, pts)
    else:
        def transform(pts):
            p_cam0 = _rigid(T0_pred_inv, pts)
            p_gt_cam0 = pol["s"] * (pol["R"] @ p_cam0.T).T + pol["t"]
            return _rigid(gt_poses_raw[0], p_gt_cam0)

    chunks = []
    for pv in per_view:
        mask = pv["mask"]
        if mask.ndim == 3:
            mask = mask.squeeze(-1)
        pts = pv["pts3d"][mask].astype(np.float64)
        if len(pts):
            chunks.append(transform(pts))
    world_points = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3))
    return world_points, scale, meta, pol["policy"]


def voxelize(points, bmin, voxel_size, dims):
    occupied = np.zeros(tuple(dims), dtype=bool)
    if len(points) == 0:
        return occupied
    idx = np.floor((points - bmin[None, :]) / voxel_size).astype(np.int64)
    valid = np.all((idx >= 0) & (idx < dims[None, :]), axis=1)
    idx = idx[valid]
    occupied[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occupied


def fruit_and_internode_check(occupied, bmin, voxel_m, gt, segments):
    dims = np.array(occupied.shape)
    occ_idx = np.argwhere(occupied)
    occ_world = bmin[None, :] + (occ_idx + 0.5) * voxel_m

    recon_volume = len(occ_idx) * (voxel_m ** 3)
    canopy_rel_err = (
        (recon_volume - gt["canopy_bbox_volume_m3"]) / gt["canopy_bbox_volume_m3"]
        if gt["canopy_bbox_volume_m3"]
        else None
    )

    fruit_results = []
    for fr in gt["fruit"]:
        center = np.array(fr["point"])
        gt_diam = fr["diameter_m"]
        search_r = FRUIT_SEARCH_RADIUS_FACTOR * (gt_diam / 2.0)
        d = np.linalg.norm(occ_world - center[None, :], axis=1) if len(occ_world) else np.array([])
        nearby = d[d <= search_r]
        if len(nearby) < MIN_VOXELS_FOR_FRUIT_EXTENT:
            fruit_results.append({"skipped": True})
            continue
        recon_diam = float(2.0 * nearby.max())
        fruit_results.append(
            {"skipped": False, "rel_err": (recon_diam - gt_diam) / gt_diam}
        )
    used = [r for r in fruit_results if not r["skipped"]]
    mean_fruit_rel_err = float(np.mean([abs(r["rel_err"]) for r in used])) if used else None

    rng = np.random.default_rng(0)
    shoot_segs = [s for s in segments if s["label"] == "shoot"]
    if len(shoot_segs) > MAX_INTERNODE_SEGMENTS_CHECKED:
        idx = rng.choice(len(shoot_segs), MAX_INTERNODE_SEGMENTS_CHECKED, replace=False)
        shoot_segs = [shoot_segs[i] for i in idx]

    def snap(point):
        idx = np.floor((point - bmin) / voxel_m).astype(np.int64)
        lo = np.clip(idx - INTERNODE_SNAP_TOL_VOXELS, 0, dims - 1)
        hi = np.clip(idx + INTERNODE_SNAP_TOL_VOXELS + 1, 0, dims)
        if np.any(lo >= hi):
            return None
        window = occupied[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        local = np.argwhere(window)
        if len(local) == 0:
            return None
        cand = bmin[None, :] + (lo[None, :] + local + 0.5) * voxel_m
        return cand[np.argmin(np.linalg.norm(cand - point[None, :], axis=1))]

    lengths = []
    n_checked = n_snapped = 0
    for seg in shoot_segs:
        n_checked += 1
        s0, s1 = snap(np.array(seg["p0"])), snap(np.array(seg["p1"]))
        if s0 is None or s1 is None:
            continue
        n_snapped += 1
        lengths.append(float(np.linalg.norm(s1 - s0)))

    recon_mean = float(np.mean(lengths)) if lengths else None
    internode_rel_err = (
        (recon_mean - gt["mean_internode_length_m"]) / gt["mean_internode_length_m"]
        if recon_mean is not None
        else None
    )

    return {
        "canopy_volume": {
            "ground_truth_bbox_m3": gt["canopy_bbox_volume_m3"],
            "reconstructed_occupied_voxel_volume_m3": recon_volume,
            "relative_error": canopy_rel_err,
        },
        "fruit_diameter": {
            "n_fruit_total": len(gt["fruit"]),
            "n_fruit_used": len(used),
            "mean_absolute_relative_error": mean_fruit_rel_err,
        },
        "internode_length": {
            "n_segments_checked": n_checked,
            "n_segments_snapped_both_ends": n_snapped,
            "ground_truth_mean_m": gt["mean_internode_length_m"],
            "reconstructed_mean_m": recon_mean,
            "relative_error": internode_rel_err,
        },
    }


def run_for(model, conditioning, segments, gt_scale):
    world_points, scale, _meta, policy = build_world_points(model, conditioning)

    by_res = {}
    occupied_fine = bmin_fine = None
    voxel_fine = None
    for res_name, npz_name in [("coarse_2cm", "t76_grid_coarse.npz"), ("fine_5mm", "t76_grid_fine.npz")]:
        grid = np.load(os.path.join(PHASE7_OUTPUT_DIR, npz_name))
        bmin, voxel_size = grid["bmin"], float(grid["voxel_size"])
        classical = grid["occupied"]
        dims = np.array(classical.shape)

        occ = voxelize(world_points, bmin, voxel_size, dims)
        theo, theo_robust, _ = _theoretical_recall(segments, voxel_size)
        emp, _t2, _hits = _empirical_recall(segments, occ, bmin, voxel_size)
        classical_emp, _t3, _ch = _empirical_recall(segments, classical, bmin, voxel_size)

        by_res[res_name] = {
            "voxel_size_m": voxel_size,
            "n_occupied_voxels_model": int(occ.sum()),
            "n_occupied_voxels_classical_t76": int(classical.sum()),
            "n_points_fused": int(len(world_points)),
            "theoretical_geometric_recall_one_voxel": theo,
            "empirical_reconstruction_recall_model": emp,
            "empirical_reconstruction_recall_classical_t76": classical_emp,
        }
        if res_name == "fine_5mm":
            occupied_fine, bmin_fine, voxel_fine = occ, bmin, voxel_size

    integrity = fruit_and_internode_check(
        occupied_fine, bmin_fine, voxel_fine, gt_scale, segments
    )
    return {
        "model": model,
        "conditioning": conditioning,
        "was_given_real_extrinsics": conditioning in POSE_CONDITIONED,
        "frame_policy": policy,
        "recovered_scale_factor_applied": scale,
        "by_resolution": by_res,
        "metric_scale_integrity": integrity,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()

    with open(os.path.join(PHASE7_OUTPUT_DIR, "t76_branch_segments.json")) as f:
        segments = json.load(f)
    with open(os.path.join(PHASE7_OUTPUT_DIR, "t76_ground_truth_scale.json")) as f:
        gt_scale = json.load(f)
    with open(os.path.join(PHASE7_OUTPUT_DIR, "t76_classical_baseline.json")) as f:
        t76 = json.load(f)
    with open(os.path.join(PHASE7_OUTPUT_DIR, "t77_metric_scale_integrity.json")) as f:
        t77 = json.load(f)

    results = {}
    for model in args.models:
        for scene, cond in discover(model):
            if scene != SCENE:
                continue
            key = f"{model}__{cond}"
            results[key] = run_for(model, cond, segments, gt_scale)
            fine = results[key]["by_resolution"]["fine_5mm"]
            integ = results[key]["metric_scale_integrity"]
            print(
                f"{key}: occ={fine['n_occupied_voxels_model']} "
                f"recall={fine['empirical_reconstruction_recall_model']} "
                f"fruit_relerr={integ['fruit_diameter']['mean_absolute_relative_error']} "
                f"internode_relerr={integ['internode_length']['relative_error']}"
            )

    out_path = os.path.join(output_root(REPO_DIR), "voxel_reconstruction_report.json")
    payload = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            payload = json.load(f)
    payload.setdefault("per_model_conditioning", {}).update(results)
    payload["phase7_classical_baseline"] = {
        "branch_recovery_rate_fine_grid_classical_t76": t76["branch_recovery_rate_fine_grid"],
        "fruit_recovery_rate_fine_grid_classical_t76": t76["fruit_recovery_rate_fine_grid"],
        "fruit_diameter_mean_abs_rel_err_classical_t77": t77["fruit_diameter"][
            "mean_absolute_relative_error"
        ],
        "internode_length_rel_err_classical_t77": t77["internode_length"]["relative_error"],
        "caveat": (
            "T7.6's classical baseline used real log-odds probabilistic depth fusion with "
            "free-space carving (phase4/occupancy_map.py); this script uses plain "
            "point-cloud voxelization of each model's mask-valid predicted points -- a "
            "simpler, weaker, and much denser fusion step on the model side, stated "
            "explicitly so a recall difference is not misread as purely a "
            "foundation-model difference."
        ),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
