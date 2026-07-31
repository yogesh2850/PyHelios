"""
Shared on-disk schema for the DEPTH-focused family (MoGe-2, PromptDA), plus the
real depth-error metrics both are scored with.

This is deliberately a different schema from `prediction_bundle.py`: these two
models are evaluated against real per-view ground-truth depth (Phase 1/Phase 7's
real EXR plane depth), not against full-scene reconstruction, because that is
what they actually produce. Mixing them into the reconstruction comparison
tables would compare unlike things.

Layout: output/<model>/<scene>/<conditioning>/{depth_predictions.npz, run_meta.json}

depth_predictions.npz, per view i:
    pred_depth__i  (H, W) float32 -- predicted metric plane depth, metres
    gt_depth__i    (H, W) float32 -- real Helios EXR plane depth resampled onto
                   the model's own output grid with INTER_NEAREST (never
                   bilinear: interpolating across this canopy's depth
                   discontinuities would invent depths that exist on neither
                   surface, which is exactly the artifact Phase 1's own noise
                   model was written to simulate rather than something to
                   accidentally bake into ground truth)
    valid__i       (H, W) bool    -- real ray hit; Helios' sky/no-hit sentinel
                   is exactly -1.0, so valid == gt_depth > 0
"""

import json
import os

import numpy as np


def run_dir(repo_dir, model, scene, conditioning):
    return os.path.join(repo_dir, "output", model, scene, conditioning)


def save_depth_bundle(repo_dir, model, scene, conditioning, per_view, meta):
    d = run_dir(repo_dir, model, scene, conditioning)
    os.makedirs(d, exist_ok=True)
    payload = {}
    for i, pv in enumerate(per_view):
        payload[f"pred_depth__{i}"] = np.asarray(pv["pred_depth"], dtype=np.float32)
        payload[f"gt_depth__{i}"] = np.asarray(pv["gt_depth"], dtype=np.float32)
        payload[f"valid__{i}"] = np.asarray(pv["valid"], dtype=bool)
    np.savez_compressed(os.path.join(d, "depth_predictions.npz"), **payload)
    with open(os.path.join(d, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return d


def load_depth_bundle(repo_dir, model, scene, conditioning):
    d = run_dir(repo_dir, model, scene, conditioning)
    with open(os.path.join(d, "run_meta.json")) as f:
        meta = json.load(f)
    npz = np.load(os.path.join(d, "depth_predictions.npz"))
    per_view = [
        {
            "pred_depth": npz[f"pred_depth__{i}"],
            "gt_depth": npz[f"gt_depth__{i}"],
            "valid": npz[f"valid__{i}"],
        }
        for i in range(meta["n_views"])
    ]
    return meta, per_view


def depth_metrics(pred, gt, valid):
    """Standard monocular-depth metrics over real, valid (ray-hit) pixels only.

    Returns absolute metrics AND median-scale-aligned ones. Both are reported
    because a metric-depth model getting the SHAPE right but the global SCALE
    wrong is a materially different failure from getting both wrong, and
    reporting only one of the two hides which happened.
    """
    p = pred[valid].astype(np.float64)
    g = gt[valid].astype(np.float64)
    keep = np.isfinite(p) & np.isfinite(g) & (g > 0)
    p, g = p[keep], g[keep]
    if p.size == 0:
        return {"n_valid_pixels": 0}

    def _core(pp):
        err = pp - g
        return {
            "rmse_mm": float(np.sqrt(np.mean(err ** 2))) * 1000.0,
            "mae_mm": float(np.mean(np.abs(err))) * 1000.0,
            "median_abs_err_mm": float(np.median(np.abs(err))) * 1000.0,
            "abs_rel": float(np.mean(np.abs(err) / g)),
            "delta1": float(np.mean(np.maximum(pp / g, g / pp) < 1.25)),
            "delta2": float(np.mean(np.maximum(pp / g, g / pp) < 1.25 ** 2)),
        }

    scale = float(np.median(g) / np.median(p)) if np.median(p) > 0 else 1.0
    out = {"n_valid_pixels": int(p.size), "median_scale_factor_pred_to_gt": scale}
    out.update(_core(p))
    out.update({f"scaled_{k}": v for k, v in _core(p * scale).items()})
    return out


def aggregate_metrics(per_view):
    """Pool every valid pixel across all views into one set of metrics, and also
    report the per-view spread so a single bad view is visible rather than
    averaged away."""
    preds = np.concatenate([pv["pred_depth"][pv["valid"]].ravel() for pv in per_view])
    gts = np.concatenate([pv["gt_depth"][pv["valid"]].ravel() for pv in per_view])
    pooled = depth_metrics(preds, gts, np.ones_like(gts, dtype=bool))

    per_view_rmse = []
    for pv in per_view:
        m = depth_metrics(pv["pred_depth"], pv["gt_depth"], pv["valid"])
        if m.get("n_valid_pixels"):
            per_view_rmse.append(m["rmse_mm"])
    pooled["per_view_rmse_mm_mean"] = float(np.mean(per_view_rmse)) if per_view_rmse else None
    pooled["per_view_rmse_mm_min"] = float(np.min(per_view_rmse)) if per_view_rmse else None
    pooled["per_view_rmse_mm_max"] = float(np.max(per_view_rmse)) if per_view_rmse else None
    pooled["n_views_scored"] = len(per_view_rmse)
    return pooled
