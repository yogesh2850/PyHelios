"""
Model-agnostic loader for Phase 7's real WAI-format scenes (real Helios render:
RGB JPEG + EXR plane depth + per-frame pose/intrinsics in scene_meta.json).

Deliberately a near-copy of `yogesh_dev/mapanything_eval/wai_loader.py`'s real
reading logic (same `OpenEXR.File(path).channels()["Z"].pixels` convention that
Phase 7's own `render_utils.py:_read_depth_exr` used, same -1.0 sky sentinel)
so every foundation model in this directory sees byte-identical inputs to what
MapAnything saw -- that identity is the whole point of the side-by-side
comparison. The MapAnything-specific view-dict builder is NOT copied; each
model's own `run_*.py` adapts these frames to its own API.

WAI conventions (from scene_meta.json, written by yogesh_dev/phase7/wai_writer.py):
  - camera_convention: "opencv" (X-right, Y-down, Z-forward)
  - transform_matrix: 4x4 camera-to-world
  - world_to_camera_matrix: 4x4 world-to-camera (its inverse, stored explicitly)
  - depth EXR: PLANE depth (distance along camera forward/Z axis), sky/no-hit
    sentinel exactly -1.0
  - intrinsics shared across all frames in a scene: fl_x, fl_y, cx, cy, w, h
"""

import json
import os

import numpy as np
import OpenEXR
from PIL import Image

SCENES = ["tree0_orbit16", "tree_t76_mvs_rig"]


def phase7_wai_root(repo_dir):
    return os.path.join(
        repo_dir, "..", "phase7", "output", "wai_dataset", "helios_apple_tree"
    )


def _read_depth_exr(filepath):
    f = OpenEXR.File(filepath)
    channels = f.channels()
    key = "Z" if "Z" in channels else next(iter(channels.keys()))
    return np.array(channels[key].pixels, dtype=np.float32)


def load_scene_meta(scene_dir):
    with open(os.path.join(scene_dir, "scene_meta.json")) as f:
        return json.load(f)


def intrinsics_matrix(meta):
    return np.array(
        [
            [meta["fl_x"], 0.0, meta["cx"]],
            [0.0, meta["fl_y"], meta["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def load_wai_frames(scene_dir, stride=1, load_depth=True):
    """Returns (frames, meta). Each frame dict holds real arrays read from disk:
    {frame_name, img_path, img (H,W,3 uint8), depth_z (H,W float32 plane depth,
     -1.0 = sky), pose_c2w (4,4), pose_w2c (4,4), K (3,3), valid_depth_mask (H,W bool)}.

    `pose_w2c` is read from the scene's own persisted `world_to_camera_matrix`
    rather than inverting `transform_matrix`, so models that want w2c
    extrinsics (DA3, VGGT) get exactly the matrix Phase 7 wrote, not a
    re-derived one.
    """
    meta = load_scene_meta(scene_dir)
    K = intrinsics_matrix(meta)
    frames = meta["frames"][::stride]
    out = []
    for fr in frames:
        img_path = os.path.join(scene_dir, fr["file_path"])
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        pose_c2w = np.array(fr["transform_matrix"], dtype=np.float64).reshape(4, 4)
        pose_w2c = np.array(fr["world_to_camera_matrix"], dtype=np.float64).reshape(4, 4)
        entry = {
            "frame_name": fr["frame_name"],
            "img_path": img_path,
            "img": img,
            "pose_c2w": pose_c2w,
            "pose_w2c": pose_w2c,
            "K": K,
        }
        if load_depth:
            depth = _read_depth_exr(os.path.join(scene_dir, fr["depth_path"]))
            entry["depth_z"] = depth
            entry["valid_depth_mask"] = depth > 0.0  # sky/no-hit sentinel is exactly -1.0
        out.append(entry)
    assert img.shape[0] == meta["h"] and img.shape[1] == meta["w"], (
        f"image size {img.shape[:2]} != scene_meta w/h {(meta['w'], meta['h'])}"
    )
    return out, meta
