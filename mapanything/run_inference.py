"""
Run real MapAnything inference on a real Phase 7 WAI scene at a given
conditioning level, save raw predictions (npz, one per scene/condition) and
timing, so `compare_ground_truth.py` / `compare_pose_accuracy.py` can consume
them without re-running the model.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
MAPANYTHING_DIR = os.path.join(REPO_DIR, "map-anything")
sys.path.insert(0, MAPANYTHING_DIR)

from mapanything.models import MapAnything  # noqa: E402
from mapanything.utils.image import preprocess_inputs  # noqa: E402

from wai_loader import load_wai_frames, frames_to_mapanything_views  # noqa: E402

OUTPUT_DIR = os.path.join(REPO_DIR, "output")
PHASE7_WAI_ROOT = os.path.join(
    REPO_DIR, "..", "phase7", "output", "wai_dataset", "helios_apple_tree"
)

_MODEL_CACHE = {}


def get_model(device, model_name="facebook/map-anything"):
    if model_name not in _MODEL_CACHE:
        t0 = time.time()
        model = MapAnything.from_pretrained(model_name).to(device)
        model.eval()
        load_s = time.time() - t0
        _MODEL_CACHE[model_name] = (model, load_s)
    return _MODEL_CACHE[model_name]


def run_one(scene_name, conditioning, device="cuda", model_name="facebook/map-anything"):
    scene_dir = os.path.join(PHASE7_WAI_ROOT, scene_name)
    frames, meta = load_wai_frames(scene_dir)
    n_views = len(frames)

    model, model_load_s = get_model(device, model_name)

    raw_views = frames_to_mapanything_views(frames, conditioning)

    t_pre0 = time.time()
    processed = preprocess_inputs(raw_views)
    # preprocess_inputs returns numpy/torch mixed; convert to torch batched (B=1) tensors
    # per mapanything's own convention (each field has a leading batch dim).
    for v in processed:
        for k, val in list(v.items()):
            if isinstance(val, np.ndarray):
                v[k] = torch.from_numpy(val)
            if torch.is_tensor(v[k]) and v[k].dim() >= 2 and k in (
                "img", "intrinsics", "depth_z", "camera_poses"
            ):
                if v[k].shape[0] != 1:
                    v[k] = v[k].unsqueeze(0)
            if k == "is_metric_scale" and not torch.is_tensor(v[k]):
                v[k] = torch.tensor([bool(v[k])])
    pre_s = time.time() - t_pre0

    if device == "cuda":
        torch.cuda.synchronize()
    t_infer0 = time.time()
    with torch.no_grad():
        outputs = model.infer(
            processed,
            memory_efficient_inference=True,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    infer_s = time.time() - t_infer0

    # Pull results to CPU numpy for saving / downstream analysis (no torch
    # dependency needed in compare_*.py, and this is small enough at these
    # scene sizes -- 16/42 views at ~500x400 -- to keep fully in memory).
    per_view = []
    for pred in outputs:
        per_view.append(
            {
                "pts3d": pred["pts3d"][0].detach().float().cpu().numpy(),
                "depth_z": pred["depth_z"][0].detach().float().cpu().numpy(),
                "intrinsics": pred["intrinsics"][0].detach().float().cpu().numpy(),
                "camera_poses": pred["camera_poses"][0].detach().float().cpu().numpy(),
                "mask": pred["mask"][0].detach().cpu().numpy().astype(bool),
                "conf": pred["conf"][0].detach().float().cpu().numpy(),
                "metric_scaling_factor": float(pred["metric_scaling_factor"][0].detach().cpu().item()),
            }
        )

    result = {
        "scene_name": scene_name,
        "conditioning": conditioning,
        "n_views": n_views,
        "model_name": model_name,
        "device": device,
        "model_load_s": model_load_s,
        "preprocess_s": pre_s,
        "infer_s": infer_s,
        "infer_s_per_view": infer_s / n_views,
        "frame_names": [fr["frame_name"] for fr in frames],
        "gt_pose_c2w": [fr["pose_c2w"].tolist() for fr in frames],
        "gt_K": frames[0]["K"].tolist(),
        "img_hw_original": [meta["h"], meta["w"]],
    }

    out_dir = os.path.join(OUTPUT_DIR, scene_name, conditioning)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(
        os.path.join(out_dir, "predictions.npz"),
        **{f"{k}__{i}": v for i, pv in enumerate(per_view) for k, v in pv.items()},
    )
    print(
        f"[{scene_name}/{conditioning}] n_views={n_views} "
        f"model_load_s={model_load_s:.2f} preprocess_s={pre_s:.3f} "
        f"infer_s={infer_s:.3f} ({infer_s / n_views:.4f} s/view)"
    )
    return result, per_view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=["tree0_orbit16", "tree_t76_mvs_rig"])
    ap.add_argument("--conditioning", required=True, choices=["images_only", "full"])
    ap.add_argument("--model_name", default="facebook/map-anything")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_one(args.scene, args.conditioning, device=device, model_name=args.model_name)


if __name__ == "__main__":
    main()
