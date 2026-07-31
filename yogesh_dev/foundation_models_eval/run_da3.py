"""
Real Depth Anything 3 (ByteDance-Seed/Depth-Anything-3) inference on Phase 7's
real WAI apple-tree scenes, saved into the shared prediction-bundle schema
(`prediction_bundle.py`) so the same compare_*.py scripts that scored
MapAnything score DA3 too.

Conditioning levels (what DA3 architecturally supports, verified by reading
`src/depth_anything_3/api.py:DepthAnything3.inference`, not guessed):
  - `images_only`      : images only, camera decoder head for pose
  - `images_only_ray`  : images only, `use_ray_pose=True` (DA3's ray head --
                         the mode the design doc specifically cares about;
                         README's own FAQ reports it as slower but more
                         accurate for pose)
  - `full`             : images + real intrinsics (3,3) + real extrinsics
                         (4,4 world-to-camera) -- DA3's pose-conditioned mode

DA3 has NO depth-conditioning input (`inference()` takes only `extrinsics` and
`intrinsics`), so unlike MapAnything's "full" level there is no `depth_z` arm
here. That is an architectural difference, not an omission -- stated plainly
in LOG.md rather than silently equating the two "full" levels.

Frame convention: DA3's `extrinsics` are WORLD-TO-CAMERA both on input and
output (confirmed in-repo: `utils/export/colmap.py` annotates
`prediction.extrinsics` with `# w2c`, and `api.py:_normalize_extrinsics`
computes `c2ws = affine_inverse(ex_t_norm)`). Phase 7's WAI scenes persist
their own `world_to_camera_matrix` per frame, which is exactly this -- fed
through unchanged, no convention conversion invented.

Validity mask: DA3's OWN default point-export rule, copied exactly from
`utils/export/glb.py:get_conf_thresh` / `export_to_glb` defaults
(conf_thresh=1.05 clipped into [percentile(conf,40), percentile(conf,90)],
then conf > threshold), plus its own predicted sky mask when available. This
is the validity signal available at inference time -- never ground-truth
depth -- matching the deliberately real-world-honest choice the MapAnything
eval made.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "Depth-Anything-3", "src"))

from depth_anything_3.api import DepthAnything3  # noqa: E402

from prediction_bundle import as_4x4, save_bundle, unproject_depth_to_world  # noqa: E402
from wai_loader import load_wai_frames, phase7_wai_root  # noqa: E402

CONDITIONINGS = ["images_only", "images_only_ray", "full"]

# DA3's own export_to_glb defaults (src/depth_anything_3/utils/export/glb.py).
GLB_CONF_THRESH = 1.05
GLB_CONF_LOWER_PCT = 40.0
GLB_CONF_UPPER_PCT = 90.0

_MODEL_CACHE = {}


def get_model(hf_id, device):
    if hf_id not in _MODEL_CACHE:
        t0 = time.time()
        model = DepthAnything3.from_pretrained(hf_id).to(device)
        model.eval()
        _MODEL_CACHE[hf_id] = (model, time.time() - t0)
    return _MODEL_CACHE[hf_id]


def da3_validity_mask(conf, sky):
    """DA3's own default confidence rule, copied from its glb exporter."""
    sky_bool = None
    if sky is not None:
        sky_bool = np.asarray(sky) > 0.5
    if sky_bool is not None and (~sky_bool).sum() > 10:
        conf_pixels = conf[~sky_bool]
    else:
        conf_pixels = conf
    lower = np.percentile(conf_pixels, GLB_CONF_LOWER_PCT)
    upper = np.percentile(conf_pixels, GLB_CONF_UPPER_PCT)
    thr = min(max(GLB_CONF_THRESH, lower), upper)
    mask = conf > thr
    if sky_bool is not None:
        mask = mask & (~sky_bool)
    return mask, float(thr)


def run_one(scene_name, conditioning, hf_id, model_tag, device="cuda"):
    scene_dir = os.path.join(phase7_wai_root(REPO_DIR), scene_name)
    frames, meta = load_wai_frames(scene_dir, load_depth=False)
    n_views = len(frames)

    model, model_load_s = get_model(hf_id, device)

    image_paths = [fr["img_path"] for fr in frames]
    if conditioning == "full":
        extrinsics = np.stack([fr["pose_w2c"] for fr in frames]).astype(np.float32)
        intrinsics = np.stack([fr["K"] for fr in frames]).astype(np.float32)
    else:
        extrinsics, intrinsics = None, None
    use_ray_pose = conditioning == "images_only_ray"

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    prediction = model.inference(
        image_paths,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        use_ray_pose=use_ray_pose,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    infer_s = time.time() - t0

    depth = np.asarray(prediction.depth)            # (N, H, W)
    conf = np.asarray(prediction.conf)              # (N, H, W)
    pred_K = np.asarray(prediction.intrinsics)      # (N, 3, 3) at DA3's output resolution
    pred_w2c = prediction.extrinsics                # (N, 4, 4) or (N, 3, 4) when pose-conditioned
    sky = prediction.sky

    per_view = []
    conf_thresholds = []
    for i in range(n_views):
        w2c = as_4x4(pred_w2c[i])
        c2w = np.linalg.inv(w2c)
        mask, thr = da3_validity_mask(conf[i], None if sky is None else sky[i])
        mask = mask & (depth[i] > 0)
        conf_thresholds.append(thr)
        per_view.append(
            {
                "pts3d": unproject_depth_to_world(depth[i].astype(np.float64), pred_K[i], c2w),
                "mask": mask,
                "intrinsics": pred_K[i],
                "camera_poses": c2w,
            }
        )

    out_meta = {
        "model": f"da3_{model_tag}",
        "model_hf_id": hf_id,
        "scene_name": scene_name,
        "conditioning": conditioning,
        "use_ray_pose": use_ray_pose,
        "n_views": n_views,
        "device": device,
        "model_load_s": model_load_s,
        "infer_s": infer_s,
        "infer_s_per_view": infer_s / n_views,
        "pred_hw": [int(depth.shape[1]), int(depth.shape[2])],
        "img_hw_original": [meta["h"], meta["w"]],
        "frame_names": [fr["frame_name"] for fr in frames],
        "gt_pose_c2w": [fr["pose_c2w"].tolist() for fr in frames],
        "gt_K": frames[0]["K"].tolist(),
        "is_metric_flag": int(getattr(prediction, "is_metric", 0) or 0),
        "scale_factor": (
            float(prediction.scale_factor) if prediction.scale_factor is not None else None
        ),
        "conf_threshold_per_view": conf_thresholds,
        "mask_valid_fraction": float(np.mean([pv["mask"].mean() for pv in per_view])),
        "note_no_depth_conditioning": (
            "DA3's inference() accepts only extrinsics and intrinsics as conditioning; "
            "it has no depth-conditioning input, so its 'full' level is images + "
            "intrinsics + extrinsics (no GT depth), unlike MapAnything's 'full' which "
            "also ingested GT depth_z."
        ),
    }
    d = save_bundle(REPO_DIR, f"da3_{model_tag}", scene_name, conditioning, per_view, out_meta)
    print(
        f"[da3_{model_tag}/{scene_name}/{conditioning}] n_views={n_views} "
        f"pred_hw={out_meta['pred_hw']} load_s={model_load_s:.2f} "
        f"infer_s={infer_s:.3f} ({infer_s / n_views:.4f} s/view) "
        f"valid_frac={out_meta['mask_valid_fraction']:.3f} -> {d}"
    )
    return out_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_id", default="depth-anything/DA3-BASE")
    ap.add_argument("--model_tag", default="base")
    ap.add_argument("--scenes", nargs="+", default=["tree0_orbit16", "tree_t76_mvs_rig"])
    ap.add_argument("--conditionings", nargs="+", default=CONDITIONINGS)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = []
    for scene in args.scenes:
        for cond in args.conditionings:
            summary.append(run_one(scene, cond, args.hf_id, args.model_tag, device=device))

    path = os.path.join(REPO_DIR, "output", f"da3_{args.model_tag}", "run_summary.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
