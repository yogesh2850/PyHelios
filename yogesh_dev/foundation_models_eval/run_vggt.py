"""
Real VGGT (facebookresearch/vggt) inference on Phase 7's real WAI apple-tree
scenes, written into the shared prediction-bundle schema.

VGGT is included as the design doc's **comparison anchor**: it has NO
pose-conditioning input at all -- `model.aggregator(images)` takes images and
nothing else, and the camera head PREDICTS extrinsics/intrinsics rather than
accepting them. So this model runs at exactly one conditioning level,
`images_only`, and that is an architectural fact, not a skipped arm. Budgeted
less time than the pose-conditioned models, per the task's own priority order.

Checkpoint: the license-clean `facebook/VGGT-1B-Commercial` was tried FIRST and
is **gated** -- `huggingface_hub.errors.GatedRepoError: 401 ... Access to model
facebook/VGGT-1B-Commercial is restricted. You must have access to it and be
authenticated`. No unauthenticated download is possible, so the runs reported
here use `facebook/VGGT-1B`, which is NOT gated but IS non-commercial
("only the newly released checkpoint VGGT-1B-Commercial is licensed for
commercial usage -- the original checkpoint remains non-commercial", VGGT's own
README). Fine for this research evaluation, same footing as Pi3X's CC-BY-NC
weights -- but any commercial use would need the gated checkpoint, and that is
a real access blocker, recorded in LOG.md rather than glossed over.

Validity-mask sensitivity: VGGT's default confidence threshold turned out to
keep only ~0.4-5% of pixels on this canopy, far stricter than the other models,
which would confound an RMSE comparison (few, easy points score well). So the
runner takes `--conf_thresh` and both a default-threshold and a lax-threshold
arm are run, so accuracy can be separated from selectivity.

Frame convention: `pose_encoding_to_extri_intri` returns extrinsics as (B,S,3,4)
"following OpenCV convention (camera from world)" per VGGT's own README, i.e.
world-to-camera -- inverted here to the camera-to-world the shared schema wants.

3D points: built with VGGT's own `unproject_depth_map_to_point_map(depth,
extrinsic, intrinsic)`, which its README explicitly recommends over the point
head ("usually leads to more accurate 3D points than point map branch").

Validity mask: `depth_conf > 5.0`, VGGT's own scripted default
(`demo_colmap.py --conf_thres_value`, default 5.0, described there as the
"Confidence threshold value for depth filtering"). Its own confidence signal,
available at inference time -- never ground-truth depth.
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
sys.path.insert(0, os.path.join(REPO_DIR, "vggt"))

from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402

from prediction_bundle import as_4x4, save_bundle  # noqa: E402
from wai_loader import load_wai_frames, phase7_wai_root  # noqa: E402

DEPTH_CONF_THRESH = 5.0  # demo_colmap.py's own default


def run_one(scene_name, hf_id, model_tag, model, model_load_s, device="cuda",
            conf_thresh=DEPTH_CONF_THRESH, cond_name="images_only"):
    scene_dir = os.path.join(phase7_wai_root(REPO_DIR), scene_name)
    frames, meta = load_wai_frames(scene_dir, load_depth=False)
    n_views = len(frames)

    # Explicit, scene_meta-ordered path list (VGGT's loader takes the list as
    # given -- it does not re-sort -- so frame order is under our control here).
    image_paths = [fr["img_path"] for fr in frames]
    images = load_and_preprocess_images(image_paths).to(device)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            batched = images[None]
            aggregated_tokens_list, ps_idx = model.aggregator(batched)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batched.shape[-2:])
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, batched, ps_idx)
    torch.cuda.synchronize()
    infer_s = time.time() - t0

    extrinsic = extrinsic.squeeze(0).float().cpu().numpy()   # (S, 3, 4) world-to-camera
    intrinsic = intrinsic.squeeze(0).float().cpu().numpy()   # (S, 3, 3)
    depth_np = depth_map.squeeze(0).float().cpu().numpy()    # (S, H, W, 1)
    conf_np = depth_conf.squeeze(0).float().cpu().numpy()    # (S, H, W)

    world_points = unproject_depth_map_to_point_map(depth_np, extrinsic, intrinsic)  # (S,H,W,3)

    per_view = []
    for i in range(n_views):
        c2w = np.linalg.inv(as_4x4(extrinsic[i]))
        mask = (conf_np[i] > conf_thresh) & (depth_np[i, ..., 0] > 0)
        per_view.append(
            {
                "pts3d": world_points[i],
                "mask": mask,
                "intrinsics": intrinsic[i],
                "camera_poses": c2w,
            }
        )

    H, W = depth_np.shape[1], depth_np.shape[2]
    out_meta = {
        "model": model_tag,
        "model_hf_id": hf_id,
        "scene_name": scene_name,
        "conditioning": cond_name,
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
        "depth_conf_threshold": conf_thresh,
        "mask_valid_fraction": float(np.mean([pv["mask"].mean() for pv in per_view])),
        "note_no_pose_conditioning": (
            "VGGT's aggregator takes images only; its camera head predicts extrinsics and "
            "intrinsics rather than accepting them, so there is no pose- or "
            "depth-conditioned arm to run. Included per the design doc as an "
            "uncalibrated comparison anchor."
        ),
    }
    d = save_bundle(REPO_DIR, model_tag, scene_name, cond_name, per_view, out_meta)
    print(
        f"[{model_tag}/{scene_name}/{cond_name}] n_views={n_views} pred_hw={[H, W]} "
        f"load_s={model_load_s:.2f} infer_s={infer_s:.3f} ({infer_s / n_views:.4f} s/view) "
        f"valid_frac={out_meta['mask_valid_fraction']:.3f} -> {d}"
    )
    del aggregated_tokens_list, depth_map, depth_conf, pose_enc
    torch.cuda.empty_cache()
    return out_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_id", default="facebook/VGGT-1B-Commercial")
    ap.add_argument("--model_tag", default="vggt_commercial")
    ap.add_argument("--scenes", nargs="+", default=["tree0_orbit16", "tree_t76_mvs_rig"])
    ap.add_argument("--conf_thresh", type=float, default=DEPTH_CONF_THRESH)
    ap.add_argument("--cond_name", default="images_only")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = VGGT.from_pretrained(args.hf_id).to(device)
    model.eval()
    model_load_s = time.time() - t0

    summary = []
    for scene in args.scenes:
        summary.append(
            run_one(scene, args.hf_id, args.model_tag, model, model_load_s, device=device,
                    conf_thresh=args.conf_thresh, cond_name=args.cond_name)
        )

    path = os.path.join(REPO_DIR, "output", args.model_tag, f"run_summary_{args.cond_name}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
