"""
Load a Phase 7 WAI-format scene (real Helios render: RGB + EXR plane depth +
per-frame pose/intrinsics in scene_meta.json) into the view-dict format
MapAnything's `preprocess_inputs` / `model.infer` expects.

WAI conventions (from scene_meta.json, written by yogesh_dev/phase7/wai_writer.py,
looked up against the real facebookresearch/map-anything data_processing/README.md
spec in Phase 7 -- reused here, not re-derived):
  - camera_convention: "opencv" (X-right, Y-down, Z-forward)
  - transform_matrix: 4x4 camera-to-world (this is exactly what MapAnything's
    'camera_poses' input wants -- "(4,4) OpenCV cam2world", per its README)
  - depth EXR: PLANE depth (distance along camera forward/Z axis), sky/no-hit
    sentinel exactly -1.0 -- this is exactly MapAnything's 'depth_z' input
  - intrinsics shared across all frames in a scene: fl_x, fl_y, cx, cy, w, h
"""

import json
import os

import numpy as np
import OpenEXR
from PIL import Image


def _read_depth_exr(filepath):
    f = OpenEXR.File(filepath)
    channels = f.channels()
    key = "Z" if "Z" in channels else next(iter(channels.keys()))
    return np.array(channels[key].pixels, dtype=np.float32)


def load_scene_meta(scene_dir):
    with open(os.path.join(scene_dir, "scene_meta.json")) as f:
        return json.load(f)


def intrinsics_matrix(meta):
    K = np.array(
        [
            [meta["fl_x"], 0.0, meta["cx"]],
            [0.0, meta["fl_y"], meta["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return K


def load_wai_frames(scene_dir, stride=1):
    """Returns a list of dicts, one per frame, with real arrays loaded from disk:
    {frame_name, img (H,W,3 uint8), depth_z (H,W float32, plane depth, -1.0=sky),
     pose_c2w (4,4 float32), K (3,3 float32), valid_depth_mask (H,W bool)}.
    """
    meta = load_scene_meta(scene_dir)
    K = intrinsics_matrix(meta)
    frames = meta["frames"][::stride]
    out = []
    for fr in frames:
        img_path = os.path.join(scene_dir, fr["file_path"])
        depth_path = os.path.join(scene_dir, fr["depth_path"])
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        depth = _read_depth_exr(depth_path)
        pose_c2w = np.array(fr["transform_matrix"], dtype=np.float32).reshape(4, 4)
        valid = depth > 0.0  # sky/no-hit sentinel is exactly -1.0
        out.append(
            {
                "frame_name": fr["frame_name"],
                "img": img,
                "depth_z": depth,
                "pose_c2w": pose_c2w,
                "K": K,
                "valid_depth_mask": valid,
            }
        )
    assert img.shape[0] == meta["h"] and img.shape[1] == meta["w"], (
        f"image size {img.shape[:2]} != scene_meta w/h {(meta['w'], meta['h'])}"
    )
    return out, meta


def frames_to_mapanything_views(frames, conditioning):
    """conditioning: 'images_only' or 'full' (+intrinsics+extrinsics+depth).

    Returns a list of view dicts in the exact schema `preprocess_inputs`/
    `model.infer` accept (raw, un-resized -- preprocess_inputs does the
    resizing to MapAnything's working resolution internally, see README /
    mapanything/utils/image.py:preprocess_inputs).
    """
    assert conditioning in ("images_only", "full")
    views = []
    for fr in frames:
        v = {"img": fr["img"]}
        if conditioning == "full":
            v["intrinsics"] = fr["K"]
            v["depth_z"] = fr["depth_z"].copy()
            v["depth_z"][~fr["valid_depth_mask"]] = 0.0  # sky sentinel -> "no depth here"
            v["camera_poses"] = fr["pose_c2w"]
            v["is_metric_scale"] = True
        views.append(v)
    return views
