"""
Decide, EMPIRICALLY and per (model, scene, conditioning), how a model's output
frame relates to the real Helios world frame -- instead of hardcoding one
assumption per model.

This exists because the four models compared here turned out to behave three
different ways when handed real camera poses as conditioning, and assuming any
one behaviour would silently produce garbage numbers for the others:

  - **MapAnything** anchored its output world frame to view 0's own camera
    frame, but preserved the input's metric scale (the mapanything_eval found
    this the hard way: 6.5m RMSE and ~100deg rotation error until it
    re-referenced both sides to camera 0).
  - **DA3** returns the caller's input extrinsics VERBATIM under pose
    conditioning (`api.py:_align_to_input_extrinsics_intrinsics` assigns
    `prediction.extrinsics = extrinsics[..., :3, :]` and rescales depth to
    match). Verified numerically here: max |pred_c2w - gt_c2w| = 4.8e-7.
  - **Pi3X** keeps its OWN canonical frame and its OWN scale even when given
    real poses AND real metric depth -- measured recovered scale 2.17x-2.93x
    off, absolute translations nowhere near the inputs'. Its conditioning
    constrains relative geometry, not the caller's absolute gauge.

So: fit a Umeyama similarity (rotation + isotropic scale + translation) from
predicted camera centres onto real ones, after re-referencing both to camera 0.
If the recovered scale is within tolerance of 1.0 AND the residual is small,
the model preserved the caller's frame and the identity transform is used
(nothing is aligned away). Otherwise the fitted similarity is used and the
recovered scale factor is reported explicitly as part of the finding -- the
same "report the gauge, don't hide it" stance Phase 7's T7.2 took.

Both numbers are always computed and always written out, so a reader can see
the direct (unaligned) residual as well as the post-alignment one and judge for
themselves.
"""

import numpy as np

# A model that genuinely honours the caller's gauge lands essentially exactly on
# it (DA3 measured 4.8e-7 relative). A model that does not is off by tens of
# percent or more (Pi3X measured 117%-193%). Nothing observed lands in between,
# so the exact tolerance is not a knife-edge choice.
SCALE_TOL = 0.02
RESIDUAL_TOL_M = 0.05


def _centres(poses):
    return np.array([p[:3, 3] for p in poses])


def analyse(gt_poses_raw, pred_poses_raw, umeyama_alignment):
    """gt_poses_raw / pred_poses_raw: lists of (4,4) camera-to-world.

    Returns a dict with the chosen policy, the transform to take a point from
    the model's camera-0-referenced frame into the real camera-0-referenced
    frame, and both the direct and aligned residuals.
    """
    gt = [np.linalg.inv(gt_poses_raw[0]) @ p for p in gt_poses_raw]
    pred = [np.linalg.inv(pred_poses_raw[0]) @ p for p in pred_poses_raw]
    gt_c, pred_c = _centres(gt), _centres(pred)

    direct_rmse = float(np.sqrt(np.mean(np.sum((pred_c - gt_c) ** 2, axis=1))))
    align = umeyama_alignment(pred_c, gt_c, with_scale=True)

    preserved = abs(align["s"] - 1.0) <= SCALE_TOL and direct_rmse <= RESIDUAL_TOL_M
    return {
        "policy": "input_frame_preserved" if preserved else "similarity_aligned",
        "recovered_scale_factor": float(align["s"]),
        "camera_center_rmse_direct_mm": direct_rmse * 1000.0,
        "camera_center_rmse_after_similarity_alignment_mm": align["rmse"] * 1000.0,
        "R": align["R"],
        "t": align["t"],
        "s": align["s"],
        "preserved": preserved,
        "gt_poses_ref": gt,
        "pred_poses_ref": pred,
    }


def make_point_transform(policy):
    """Map a point from the model's camera-0-referenced frame into the real
    camera-0-referenced frame, per the chosen policy."""
    if policy["preserved"]:
        return lambda p: p
    R, t, s = policy["R"], policy["t"], policy["s"]

    def transform(p):
        p = np.atleast_2d(p)
        out = s * (R @ p.T).T + t
        return out

    return transform
