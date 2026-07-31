"""
Real Pi3X (yyfz/Pi3, `pi3.models.pi3x.Pi3X`) inference on Phase 7's real WAI
apple-tree scenes, written into the shared prediction-bundle schema.

Pi3X is the only model in this comparison that accepts ALL THREE conditioning
modalities MapAnything did (camera poses + intrinsics + depth), so it supports
the closest like-for-like "full" level:
  - `images_only`     : images only
  - `pose_intrinsics` : + real intrinsics + real camera-to-world poses
                        (matches DA3's most-conditioned level, which has no
                        depth input)
  - `full`            : + real metric depth as well (matches MapAnything's
                        "full")

Conventions, read out of `pi3/models/pi3x.py:Pi3X.forward`'s own docstring
(not guessed): `poses` are camera-to-world in OpenCV convention -- exactly
Phase 7's `transform_matrix` -- and `depths` want invalid pixels set to 0,
which is what Helios' -1.0 sky sentinel maps to.

## Why this file re-implements Pi3's `load_multimodal_data` instead of calling it

`pi3/utils/basic.py:load_multimodal_data` discovers images by
`sorted(os.listdir(...))`. Phase 7's frames are named `cam0_frame_00000.jpeg`,
`cam1_frame_00001.jpeg`, ..., `cam10_frame_00010.jpeg`, so that alphabetical
sort yields cam0, cam10, cam11, ... -- a DIFFERENT order from `scene_meta.json`'s
frame list, which is what the pose/depth/intrinsics condition arrays are
indexed by. Calling it directly would silently pair each image with another
frame's pose. `prepare_pi3_inputs` below reproduces its resize math exactly
(PIXEL_LIMIT=255000, LANCZOS image resize, both dims forced to multiples of 14
by the same shrink loop, intrinsics rescaled by scale_x/scale_y, depth resized
with cv2.INTER_NEAREST and non-positive values zeroed) but drives it from the
explicit scene_meta frame order. Verified against Pi3's own function; the only
intended difference is frame ordering.
"""

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "Pi3"))

from pi3.models.pi3x import Pi3X  # noqa: E402
from pi3.utils.geometry import depth_normal_edge, recover_intrinsic_from_rays_d  # noqa: E402

from prediction_bundle import save_bundle  # noqa: E402
from wai_loader import load_wai_frames, phase7_wai_root  # noqa: E402

CONDITIONINGS = ["images_only", "pose_intrinsics", "full"]
PIXEL_LIMIT = 255000  # pi3/utils/basic.py:load_multimodal_data default
CONF_SIGMOID_THRESH = 0.1  # example_mm.py's own mask rule
EDGE_RTOL = 0.03           # example_mm.py's own depth_normal_edge rtol

_MODEL_CACHE = {}


def target_size(W_orig, H_orig):
    """Pi3's own target-resolution math, reproduced exactly."""
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target:
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14


def prepare_pi3_inputs(frames, meta, conditioning, device):
    W_orig, H_orig = meta["w"], meta["h"]
    TARGET_W, TARGET_H = target_size(W_orig, H_orig)
    scale_x, scale_y = TARGET_W / W_orig, TARGET_H / H_orig

    imgs = []
    for fr in frames:
        pil = Image.open(fr["img_path"]).convert("RGB")
        pil = pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
        imgs.append(torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).permute(2, 0, 1))
    images_tensor = torch.stack(imgs, dim=0)[None].to(device)  # (1, N, 3, H, W)

    conditions = {"poses": None, "depths": None, "intrinsics": None}
    if conditioning in ("pose_intrinsics", "full"):
        K = np.stack([fr["K"].astype(np.float64) for fr in frames])
        K[:, 0, 0] *= scale_x
        K[:, 0, 2] *= scale_x
        K[:, 1, 1] *= scale_y
        K[:, 1, 2] *= scale_y
        conditions["intrinsics"] = torch.from_numpy(K).float()[None].to(device)
        poses = np.stack([fr["pose_c2w"] for fr in frames])
        conditions["poses"] = torch.from_numpy(poses).float()[None].to(device)
    if conditioning == "full":
        depths = []
        for fr in frames:
            d = cv2.resize(
                fr["depth_z"].astype(np.float32),
                (TARGET_W, TARGET_H),
                interpolation=cv2.INTER_NEAREST,
            )
            d[~np.logical_and(d > 0, np.isfinite(d))] = 0.0  # Helios sky sentinel -1.0 -> 0
            depths.append(torch.from_numpy(d))
        conditions["depths"] = torch.stack(depths, dim=0)[None].to(device)

    return images_tensor, conditions, (TARGET_H, TARGET_W)


def get_model(hf_id, use_multimodal, device):
    key = (hf_id, use_multimodal)
    if key not in _MODEL_CACHE:
        t0 = time.time()
        model = Pi3X.from_pretrained(hf_id).eval()
        if not use_multimodal:
            model.disable_multimodal()
        model = model.to(device)
        _MODEL_CACHE[key] = (model, time.time() - t0)
    return _MODEL_CACHE[key]


def run_one(scene_name, conditioning, hf_id, model_tag, device="cuda"):
    scene_dir = os.path.join(phase7_wai_root(REPO_DIR), scene_name)
    need_depth = conditioning == "full"
    frames, meta = load_wai_frames(scene_dir, load_depth=need_depth)
    n_views = len(frames)

    imgs, conditions, (H, W) = prepare_pi3_inputs(frames, meta, conditioning, device)
    use_multimodal = any(v is not None for v in conditions.values())
    model, model_load_s = get_model(hf_id, use_multimodal, device)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            res = model(imgs=imgs, **conditions)
    torch.cuda.synchronize()
    infer_s = time.time() - t0

    # Pi3's own default validity mask (example_mm.py, verbatim).
    masks = torch.sigmoid(res["conf"][..., 0]) > CONF_SIGMOID_THRESH
    non_edge = ~depth_normal_edge(res["local_points"], rtol=EDGE_RTOL, mask=masks)
    masks = torch.logical_and(masks, non_edge)[0]  # (N, H, W)

    # Pi3's own intrinsic recovery from its predicted ray directions.
    rays_d = torch.nn.functional.normalize(res["local_points"], dim=-1)
    K_pred = recover_intrinsic_from_rays_d(rays_d, force_center_principal_point=True)[0]

    points = res["points"][0].float().cpu().numpy()          # (N, H, W, 3) world
    camera_poses = res["camera_poses"][0].float().cpu().numpy()  # (N, 4, 4) cam2world
    K_pred_np = K_pred.float().cpu().numpy()                 # (N, 3, 3)
    masks_np = masks.cpu().numpy()

    per_view = [
        {
            "pts3d": points[i],
            "mask": masks_np[i],
            "intrinsics": K_pred_np[i],
            "camera_poses": camera_poses[i],
        }
        for i in range(n_views)
    ]

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
        "pred_hw": [int(H), int(W)],
        "img_hw_original": [meta["h"], meta["w"]],
        "frame_names": [fr["frame_name"] for fr in frames],
        "gt_pose_c2w": [fr["pose_c2w"].tolist() for fr in frames],
        "gt_K": frames[0]["K"].tolist(),
        "metric_scalar": float(res["metric"][0].item()) if "metric" in res else None,
        "mask_valid_fraction": float(masks_np.mean()),
        "conditions_supplied": {k: (v is not None) for k, v in conditions.items()},
    }
    d = save_bundle(REPO_DIR, model_tag, scene_name, conditioning, per_view, out_meta)
    print(
        f"[{model_tag}/{scene_name}/{conditioning}] n_views={n_views} pred_hw={[H, W]} "
        f"load_s={model_load_s:.2f} infer_s={infer_s:.3f} ({infer_s / n_views:.4f} s/view) "
        f"metric={out_meta['metric_scalar']} valid_frac={out_meta['mask_valid_fraction']:.3f} -> {d}"
    )
    del res
    torch.cuda.empty_cache()
    return out_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_id", default="yyfz233/Pi3X")
    ap.add_argument("--model_tag", default="pi3x")
    ap.add_argument("--scenes", nargs="+", default=["tree0_orbit16", "tree_t76_mvs_rig"])
    ap.add_argument("--conditionings", nargs="+", default=CONDITIONINGS)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = []
    for scene in args.scenes:
        for cond in args.conditionings:
            summary.append(run_one(scene, cond, args.hf_id, args.model_tag, device=device))

    path = os.path.join(REPO_DIR, "output", args.model_tag, "run_summary.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
