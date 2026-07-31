"""
Real MoGe-2 (microsoft/MoGe, `moge.model.v2.MoGeModel`) inference on Phase 7's
real WAI apple-tree scenes, scored against the real per-view EXR ground-truth
depth rather than against full-scene reconstruction -- MoGe-2 is a MONOCULAR
metric-geometry model, one image in, one metric point map / depth map out, so
per-view depth accuracy is what it actually claims and what it should be judged
on. Running it through the multi-view reconstruction tables would be comparing
unlike things.

Conditioning levels (what MoGe-2's `infer()` actually accepts -- verified by
reading `moge/model/v2.py:MoGeModel.infer`'s signature, not guessed; its only
optional geometric input is `fov_x`):
  - `images_only` : image only, horizontal FoV inferred by the model
  - `known_fov`   : image + the real horizontal FoV computed from Phase 7's
                    real intrinsics -- the only conditioning this architecture
                    takes. There is no pose or depth input, so there is no
                    "full" arm for MoGe-2; that is architectural, not skipped.

Depth is compared on MoGe-2's own output grid (which is the input image
resolution here, 360x480 -- no resize needed), against the real EXR plane depth
with Helios' -1.0 sky/no-hit sentinel treated as invalid, matching Phase 0/7's
own convention.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "MoGe"))

from moge.model.v2 import MoGeModel  # noqa: E402

from depth_bundle import save_depth_bundle  # noqa: E402
from wai_loader import load_wai_frames, phase7_wai_root  # noqa: E402

CONDITIONINGS = ["images_only", "known_fov"]


def run_one(scene_name, conditioning, hf_id, model_tag, model, model_load_s, device="cuda"):
    scene_dir = os.path.join(phase7_wai_root(REPO_DIR), scene_name)
    frames, meta = load_wai_frames(scene_dir, load_depth=True)
    n_views = len(frames)

    # Real horizontal FoV from the scene's real intrinsics (shared across frames).
    K = frames[0]["K"]
    fov_x_deg = float(np.degrees(2.0 * math.atan(meta["w"] / (2.0 * K[0, 0]))))

    per_view = []
    infer_times = []
    for fr in frames:
        img = torch.from_numpy(fr["img"].astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.infer(img, fov_x=(fov_x_deg if conditioning == "known_fov" else None))
        torch.cuda.synchronize()
        infer_times.append(time.time() - t0)

        pred_depth = out["depth"].float().cpu().numpy()  # (H, W) metric metres
        gt_depth = fr["depth_z"]
        valid = fr["valid_depth_mask"]
        # MoGe-2 also predicts its own validity mask; intersect it so pixels it
        # itself declines to predict are not counted against it. Its mask is the
        # signal available at inference time -- ground-truth depth is only ever
        # used as the reference, never as a gate on the model's own output.
        if "mask" in out and out["mask"] is not None:
            model_mask = out["mask"].cpu().numpy().astype(bool)
            valid = valid & model_mask
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
        "pred_hw": [int(per_view[0]["pred_depth"].shape[0]), int(per_view[0]["pred_depth"].shape[1])],
        "img_hw_original": [meta["h"], meta["w"]],
        "fov_x_deg_real": fov_x_deg,
        "fov_x_supplied": conditioning == "known_fov",
        "frame_names": [fr["frame_name"] for fr in frames],
        "valid_pixel_fraction": float(np.mean([pv["valid"].mean() for pv in per_view])),
        "note_monocular_only": (
            "MoGe-2's infer() takes one image and an optional fov_x; it has no pose or "
            "depth conditioning input, so per-view depth accuracy is the whole story for "
            "this model. Not run through the multi-view reconstruction tables."
        ),
    }
    d = save_depth_bundle(REPO_DIR, model_tag, scene_name, conditioning, per_view, out_meta)
    print(
        f"[{model_tag}/{scene_name}/{conditioning}] n_views={n_views} "
        f"infer_s={infer_s:.3f} ({infer_s / n_views:.4f} s/view) "
        f"valid_frac={out_meta['valid_pixel_fraction']:.3f} -> {d}"
    )
    return out_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_id", default="Ruicheng/moge-2-vitl-normal")
    ap.add_argument("--model_tag", default="moge2_vitl")
    ap.add_argument("--scenes", nargs="+", default=["tree0_orbit16", "tree_t76_mvs_rig"])
    ap.add_argument("--conditionings", nargs="+", default=CONDITIONINGS)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = MoGeModel.from_pretrained(args.hf_id).to(device).eval()
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
