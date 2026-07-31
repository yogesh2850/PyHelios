"""
Real per-view depth accuracy for the depth-focused family (MoGe-2, PromptDA)
against Phase 7's real EXR ground-truth depth.

Deliberately a SEPARATE report from the 3D-reconstruction tables: these models
predict per-view metric depth, not a multi-view scene reconstruction, so the
honest comparison for them is dense per-pixel depth error against the real
depth buffer -- not landmark RMSE or voxel recall. Putting them in the same
table as DA3/Pi3X/VGGT/MapAnything would invite a comparison the models do not
support.

Every metric is computed over real, valid (ray-hit) pixels only, pooled across
all views of a scene, and reported both absolutely and after median scale
alignment -- see `depth_bundle.depth_metrics` for why both.

One real classical reference point is included for context: Phase 1's own RGB-D
noise model, applied to the ground truth and then scored against that same
ground truth, gives the depth error a plausible real DEPTH SENSOR would have on
this scene. A learned model beating that number is doing better than the sensor
it is meant to clean up; missing it is doing worse. That reference is computed
here from the same real Phase 1 code, not quoted from memory.
"""

import argparse
import json
import os
import sys

import numpy as np

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE1_DIR = os.path.abspath(os.path.join(REPO_DIR, "..", "phase1"))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, PHASE1_DIR)
from noise_model import apply_rgbd_noise_model  # noqa: E402

from depth_bundle import aggregate_metrics, depth_metrics, load_depth_bundle  # noqa: E402
from wai_loader import load_wai_frames, phase7_wai_root  # noqa: E402

SCENES = ["tree0_orbit16", "tree_t76_mvs_rig"]


def discover(model):
    root = os.path.join(REPO_DIR, "output", model)
    found = []
    if not os.path.isdir(root):
        return found
    for scene in sorted(os.listdir(root)):
        sdir = os.path.join(root, scene)
        if not os.path.isdir(sdir):
            continue
        for cond in sorted(os.listdir(sdir)):
            if os.path.exists(os.path.join(sdir, cond, "depth_predictions.npz")):
                found.append((scene, cond))
    return found


def sensor_noise_reference(scene):
    """What a plausible real RGB-D sensor would score on this exact scene, using
    Phase 1's own real noise model. The floor any depth model should be judged
    against."""
    scene_dir = os.path.join(phase7_wai_root(REPO_DIR), scene)
    frames, _meta = load_wai_frames(scene_dir, load_depth=True)
    per_view = []
    for fr in frames:
        noisy = apply_rgbd_noise_model(
            fr["depth_z"].astype(np.float64), fr["valid_depth_mask"], enable=True, seed=0
        )
        per_view.append(
            {
                "pred_depth": noisy.astype(np.float32),
                "gt_depth": fr["depth_z"],
                "valid": fr["valid_depth_mask"],
            }
        )
    return aggregate_metrics(per_view)


def naive_prompt_upsampling_baseline(scene, downscale, sky_fill_m, noise=True):
    """The baseline PromptDA actually has to beat to be worth running.

    PromptDA's job is to turn a low-resolution, noisy metric depth map into a
    dense one. The trivial alternative is to just resize that same prompt back
    up with nearest-neighbour interpolation. Scoring that here, from the SAME
    real GT, the SAME Phase 1 noise model, the SAME downscale factor and the
    SAME sky-fill convention, answers "did the learned model add anything over
    plain interpolation of its own input?" -- a question the raw RMSE number
    alone cannot.
    """
    import cv2

    scene_dir = os.path.join(phase7_wai_root(REPO_DIR), scene)
    frames, meta = load_wai_frames(scene_dir, load_depth=True)
    TH, TW = int(meta["h"] // 14 * 14), int(meta["w"] // 14 * 14)
    per_view = []
    for fr in frames:
        base = (
            apply_rgbd_noise_model(
                fr["depth_z"].astype(np.float64), fr["valid_depth_mask"], enable=True, seed=0
            )
            if noise
            else fr["depth_z"].astype(np.float64)
        )
        base = np.where(fr["valid_depth_mask"], base, sky_fill_m).astype(np.float32)
        low = cv2.resize(base, (TW // downscale, TH // downscale), interpolation=cv2.INTER_NEAREST)
        up = cv2.resize(low, (TW, TH), interpolation=cv2.INTER_NEAREST)
        gt = cv2.resize(fr["depth_z"], (TW, TH), interpolation=cv2.INTER_NEAREST)
        per_view.append({"pred_depth": up, "gt_depth": gt, "valid": gt > 0.0})
    return aggregate_metrics(per_view)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--skip_sensor_reference", action="store_true")
    args = ap.parse_args()

    results = {}
    for model in args.models:
        for scene, cond in discover(model):
            meta, per_view = load_depth_bundle(REPO_DIR, model, scene, cond)
            m = aggregate_metrics(per_view)
            m.update(
                {
                    "model": model,
                    "model_hf_id": meta.get("model_hf_id"),
                    "scene_name": scene,
                    "conditioning": cond,
                    "n_views": meta["n_views"],
                    "infer_s": meta.get("infer_s"),
                    "infer_s_per_view": meta.get("infer_s_per_view"),
                    "model_load_s": meta.get("model_load_s"),
                    "pred_hw": meta.get("pred_hw"),
                    "prompt_hw": meta.get("prompt_hw"),
                }
            )
            key = f"{model}__{scene}__{cond}"
            results[key] = m
            print(
                f"{key}: rmse={m['rmse_mm']:.1f}mm mae={m['mae_mm']:.1f}mm "
                f"absrel={m['abs_rel']:.4f} d1={m['delta1']:.3f} "
                f"scaled_rmse={m['scaled_rmse_mm']:.1f}mm scale={m['median_scale_factor_pred_to_gt']:.4f} "
                f"n_px={m['n_valid_pixels']}"
            )

    out_path = os.path.join(REPO_DIR, "output", "depth_accuracy_report.json")
    payload = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            payload = json.load(f)
    payload.setdefault("per_model_scene_conditioning", {}).update(results)

    if not args.skip_sensor_reference:
        ref = payload.setdefault("phase1_sensor_noise_reference", {})
        for scene in SCENES:
            if scene not in ref:
                ref[scene] = sensor_noise_reference(scene)
                print(
                    f"[phase1 sensor-noise reference] {scene}: "
                    f"rmse={ref[scene]['rmse_mm']:.1f}mm absrel={ref[scene]['abs_rel']:.4f}"
                )
        base = payload.setdefault("naive_prompt_upsampling_baseline", {})
        for scene in SCENES:
            if scene not in base:
                base[scene] = naive_prompt_upsampling_baseline(
                    scene, downscale=4, sky_fill_m=8.0, noise=True
                )
                print(
                    f"[naive 4x nearest upsampling of the same noisy prompt] {scene}: "
                    f"rmse={base[scene]['rmse_mm']:.1f}mm absrel={base[scene]['abs_rel']:.4f}"
                )
        payload["naive_prompt_upsampling_baseline_note"] = (
            "Same real GT, same Phase 1 noise model (seed 0), same 4x downscale, same "
            "8.0m background sky fill as PromptDA's `prompt_sparse_noisy_bgfill` arm -- "
            "then simply resized back up with INTER_NEAREST instead of run through "
            "PromptDA. This is the do-nothing alternative PromptDA has to beat."
        )
        payload["phase1_sensor_noise_reference_note"] = (
            "Phase 1's real RGB-D noise model (yogesh_dev/phase1/noise_model.py) applied to "
            "the real GT depth and scored against that same GT -- i.e. the depth error a "
            "plausible RealSense/ZED-class sensor would itself have on this exact canopy. "
            "Not a model result; a reference floor."
        )

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
