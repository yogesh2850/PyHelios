"""
One common on-disk prediction schema shared by every 3D-reconstruction model
evaluated here (DA3, Pi3X, VGGT), byte-compatible with the schema
`yogesh_dev/mapanything_eval/` already wrote for MapAnything.

That compatibility is deliberate and load-bearing: it means the SAME
`compare_pose_accuracy.py` / `compare_landmark_rmse.py` /
`compare_voxel_reconstruction.py` code paths score every model, so a number
difference between models is a real model difference and not a difference in
how each was scored.

Layout: output/<model>/<scene>/<conditioning>/{predictions.npz, run_meta.json}

predictions.npz, per view i (0..n_views-1):
    pts3d__i        (H, W, 3) float32 -- world-frame 3D point per output pixel,
                    in whatever frame the model natively predicts in
    mask__i         (H, W)    bool    -- the model's OWN validity/confidence mask,
                    i.e. the signal actually available at inference time (never
                    ground-truth depth)
    intrinsics__i   (3, 3)   float32 -- the model's own K at ITS output
                    resolution (H, W above), not the original image K
    camera_poses__i (4, 4)   float32 -- camera-to-world, OpenCV convention

run_meta.json: scene/conditioning/model/device, n_views, timings, frame_names,
gt_pose_c2w (real Helios cam2world per frame), gt_K, img_hw_original, plus any
model-specific notes.
"""

import json
import os

import numpy as np


def output_root(repo_dir):
    return os.path.join(repo_dir, "output")


def run_dir(repo_dir, model, scene, conditioning):
    return os.path.join(output_root(repo_dir), model, scene, conditioning)


def save_bundle(repo_dir, model, scene, conditioning, per_view, meta):
    """per_view: list of dicts with keys pts3d, mask, intrinsics, camera_poses."""
    d = run_dir(repo_dir, model, scene, conditioning)
    os.makedirs(d, exist_ok=True)
    payload = {}
    for i, pv in enumerate(per_view):
        payload[f"pts3d__{i}"] = np.asarray(pv["pts3d"], dtype=np.float32)
        payload[f"mask__{i}"] = np.asarray(pv["mask"], dtype=bool)
        payload[f"intrinsics__{i}"] = np.asarray(pv["intrinsics"], dtype=np.float32)
        payload[f"camera_poses__{i}"] = np.asarray(pv["camera_poses"], dtype=np.float32)
    np.savez_compressed(os.path.join(d, "predictions.npz"), **payload)
    with open(os.path.join(d, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return d


def load_bundle(repo_dir, model, scene, conditioning):
    d = run_dir(repo_dir, model, scene, conditioning)
    with open(os.path.join(d, "run_meta.json")) as f:
        meta = json.load(f)
    npz = np.load(os.path.join(d, "predictions.npz"))
    per_view = [
        {
            "pts3d": npz[f"pts3d__{i}"],
            "mask": npz[f"mask__{i}"],
            "intrinsics": npz[f"intrinsics__{i}"],
            "camera_poses": npz[f"camera_poses__{i}"],
        }
        for i in range(meta["n_views"])
    ]
    return meta, per_view


def unproject_depth_to_world(depth, K, pose_c2w):
    """(H,W) plane depth + (3,3) K at that resolution + (4,4) cam2world -> (H,W,3)
    world points. OpenCV pinhole convention (x right, y down, z forward), the
    convention every model here and the WAI dataset itself use."""
    H, W = depth.shape
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    x_cam = (xs - K[0, 2]) / K[0, 0] * depth
    y_cam = (ys - K[1, 2]) / K[1, 1] * depth
    pts_cam = np.stack([x_cam, y_cam, depth], axis=-1)
    return pts_cam @ pose_c2w[:3, :3].T + pose_c2w[:3, 3]


def as_4x4(T):
    """Accept (3,4) or (4,4) and always return (4,4). DA3 returns (N,3,4)
    extrinsics on the pose-conditioned code path and (N,4,4) otherwise."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape == (4, 4):
        return T
    if T.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :] = T
        return out
    raise ValueError(f"unexpected transform shape {T.shape}")
