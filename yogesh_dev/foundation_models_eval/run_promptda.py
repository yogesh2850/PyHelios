"""
Real PromptDA / Prompt Depth Anything (DepthAnything/PromptDA, CVPR 2025)
inference on Phase 7's real WAI apple-tree scenes.

PromptDA takes RGB **plus a low-resolution, noisy metric depth "prompt"**
(designed around iPhone ARKit LiDAR) and outputs dense metric depth. So it
needs a plausible *sensor-like* depth input, which this script builds from the
real ground-truth EXR depth in the order a real sensor would produce it:

    real dense GT depth (metres, -1.0 = sky/no-hit)
      -> Phase 1's REAL RGB-D noise model, at full resolution
         (`yogesh_dev/phase1/noise_model.py:apply_rgbd_noise_model`, imported
         and called directly -- NOT reimplemented here: it injects flying/mixed
         pixels at genuine depth discontinuities and range-dependent noise
         sigma(z) = 0.0015 * z^2, both tuned in Phase 1 against this exact dense
         canopy geometry)
      -> low-resolution readout (INTER_NEAREST subsample by PROMPT_DOWNSCALE)

Noise first, then subsample, because that is the physical order: a stereo/LiDAR
sensor's mixed pixels and range noise happen at the sensing surface, and the
low-resolution depth map is the readout of that already-noisy measurement.
Doing it the other way round would filter the mixed-pixel artifact away with
the subsampling and quietly make the task easier than reality.

Conditioning levels:
  - `prompt_sparse_noisy` : the realistic one -- subsampled AND noise-injected
  - `prompt_sparse_clean` : subsampled only, no noise. Not realistic; included
                            purely to isolate how much of the error is the
                            noise vs the sparsity.
  - `prompt_dense_noisy`  : noise-injected at full resolution, no subsampling.
                            Isolates sparsity's contribution the other way.

Ground truth for scoring is always the REAL, CLEAN, DENSE EXR depth -- the
noisy/sparse version is only ever an input, never the reference.

## Resolution

PromptDA is a ViT with patch size 14 and its own loader floors each image
dimension to a multiple of 14 (`promptda/utils/io_wrapper.py:load_image` /
`ensure_multiple_of`). Phase 7's frames are 480x360, neither of which is a
multiple of 14, so this script resizes to 476x350 (34x14, 25x14) with
INTER_AREA for RGB and INTER_NEAREST for depth, and scores against GT depth
resampled to that same 476x350 grid with INTER_NEAREST. Nearest is used for all
depth resampling so no depth value is ever invented between two surfaces
straddling a canopy occlusion boundary.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE1_DIR = os.path.abspath(os.path.join(REPO_DIR, "..", "phase1"))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, PHASE1_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "PromptDA"))

from noise_model import apply_rgbd_noise_model  # noqa: E402  (Phase 1's real model)
from promptda.promptda import PromptDA  # noqa: E402

from depth_bundle import save_depth_bundle  # noqa: E402
from wai_loader import load_wai_frames, phase7_wai_root  # noqa: E402

CONDITIONINGS = [
    "prompt_sparse_noisy",
    "prompt_sparse_clean",
    "prompt_dense_noisy",
    "prompt_sparse_noisy_bgfill",
]
PATCH_MULTIPLE = 14
# ~4x, matching the ratio PromptDA's own example ships with (1008x756 image
# against a 256x192 ARKit LiDAR depth map is 3.94x).
PROMPT_DOWNSCALE = 4
NOISE_SEED = 0

# Only 6.04% of pixels in these scenes are a real ray hit -- the other ~94% is
# open sky behind a thin canopy, where Helios writes its -1.0 no-hit sentinel.
# How that vast invalid region is presented to PromptDA turns out to dominate
# its output, so both options are run rather than one being assumed:
#
#   SKY_FILL_ZERO       : sky -> 0.0, which is the invalid convention PromptDA's
#                         own ARKit data uses (its `load_depth` reads a uint16
#                         PNG /1000, where 0 means no return). Realistic for a
#                         LiDAR pointed at open sky.
#   SKY_FILL_BACKGROUND : sky -> a constant assumed background distance. A real
#                         orchard row has ground and a next tree row behind the
#                         canopy at roughly this range; Helios simply rendered
#                         nothing there. This is an explicit modelling
#                         assumption, stated as such, not a measurement.
BACKGROUND_DEPTH_M = 8.0


def _floor_to_multiple(x, m=PATCH_MULTIPLE):
    return int(x // m * m)


def build_inputs(fr, meta, conditioning, target_hw):
    TH, TW = target_hw
    img = cv2.resize(fr["img"], (TW, TH), interpolation=cv2.INTER_AREA)
    img_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None]

    # Ground truth for SCORING: the real, clean, dense EXR depth on the model's grid.
    gt_depth = cv2.resize(fr["depth_z"], (TW, TH), interpolation=cv2.INTER_NEAREST)
    valid = gt_depth > 0.0  # Helios sky/no-hit sentinel is exactly -1.0

    # Sensor-like depth for the PROMPT input, built from the real GT.
    if conditioning != "prompt_sparse_clean":
        noisy_full = apply_rgbd_noise_model(
            fr["depth_z"].astype(np.float64),
            fr["valid_depth_mask"],
            enable=True,
            seed=NOISE_SEED,
        )
    else:
        noisy_full = fr["depth_z"].astype(np.float64).copy()
    # Invalid (sky) pixels carry no measurement; leaving Helios' -1.0 sentinel in
    # would corrupt PromptDA's own min/max prompt normalisation.
    sky_value = BACKGROUND_DEPTH_M if conditioning.endswith("_bgfill") else 0.0
    noisy_full = np.where(fr["valid_depth_mask"], noisy_full, sky_value).astype(np.float32)

    if conditioning == "prompt_dense_noisy":
        prompt = cv2.resize(noisy_full, (TW, TH), interpolation=cv2.INTER_NEAREST)
    else:
        pw, ph = TW // PROMPT_DOWNSCALE, TH // PROMPT_DOWNSCALE
        prompt = cv2.resize(noisy_full, (pw, ph), interpolation=cv2.INTER_NEAREST)
    prompt_t = torch.from_numpy(prompt.astype(np.float32))[None, None]

    return img_t, prompt_t, gt_depth, valid


def run_one(scene_name, conditioning, hf_id, model_tag, model, model_load_s, device="cuda"):
    scene_dir = os.path.join(phase7_wai_root(REPO_DIR), scene_name)
    frames, meta = load_wai_frames(scene_dir, load_depth=True)
    n_views = len(frames)
    target_hw = (_floor_to_multiple(meta["h"]), _floor_to_multiple(meta["w"]))

    per_view = []
    infer_times = []
    prompt_hw = None
    for fr in frames:
        img_t, prompt_t, gt_depth, valid = build_inputs(fr, meta, conditioning, target_hw)
        prompt_hw = list(prompt_t.shape[-2:])
        img_t, prompt_t = img_t.to(device), prompt_t.to(device)

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            pred = model.predict(img_t, prompt_t)
        torch.cuda.synchronize()
        infer_times.append(time.time() - t0)

        pred_depth = pred.squeeze().float().cpu().numpy()
        if pred_depth.shape != gt_depth.shape:
            pred_depth = cv2.resize(
                pred_depth, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        per_view.append({"pred_depth": pred_depth, "gt_depth": gt_depth, "valid": valid})

    infer_s = float(np.sum(infer_times))
    out_meta = {
        "model": model_tag,
        "model_hf_id": hf_id,
        "scene_name": scene_name,
        "conditioning": conditioning,
        "n_views": n_views,
        "device": device,
        "model_load_s": model_load_s,
        "infer_s": infer_s,
        "infer_s_per_view": infer_s / n_views,
        "pred_hw": list(target_hw),
        "prompt_hw": prompt_hw,
        "prompt_downscale": 1 if conditioning == "prompt_dense_noisy" else PROMPT_DOWNSCALE,
        "noise_injected": conditioning != "prompt_sparse_clean",
        "sky_fill_value_m": BACKGROUND_DEPTH_M if conditioning.endswith("_bgfill") else 0.0,
        "noise_model_source": (
            "yogesh_dev/phase1/noise_model.py:apply_rgbd_noise_model (Phase 1's real "
            "RGB-D noise model: mixed/flying pixels at depth discontinuities + "
            "range-dependent sigma(z)=0.0015*z^2), imported and called directly, "
            f"seed={NOISE_SEED}, applied at full resolution BEFORE low-res readout"
        ),
        "img_hw_original": [meta["h"], meta["w"]],
        "frame_names": [fr["frame_name"] for fr in frames],
        "valid_pixel_fraction": float(np.mean([pv["valid"].mean() for pv in per_view])),
    }
    d = save_depth_bundle(REPO_DIR, model_tag, scene_name, conditioning, per_view, out_meta)
    print(
        f"[{model_tag}/{scene_name}/{conditioning}] n_views={n_views} "
        f"pred_hw={target_hw} prompt_hw={prompt_hw} "
        f"infer_s={infer_s:.3f} ({infer_s / n_views:.4f} s/view) -> {d}"
    )
    return out_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_id", default="depth-anything/prompt-depth-anything-vitl")
    ap.add_argument("--model_tag", default="promptda_vitl")
    ap.add_argument("--scenes", nargs="+", default=["tree0_orbit16", "tree_t76_mvs_rig"])
    ap.add_argument("--conditionings", nargs="+", default=CONDITIONINGS)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = PromptDA.from_pretrained(args.hf_id).to(device).eval()
    model_load_s = time.time() - t0

    summary = []
    for scene in args.scenes:
        for cond in args.conditionings:
            summary.append(
                run_one(scene, cond, args.hf_id, args.model_tag, model, model_load_s, device=device)
            )

    path = os.path.join(REPO_DIR, "output", args.model_tag, "run_summary.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
