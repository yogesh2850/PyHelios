# MapAnything eval — log

Working directory: isolated git worktree `.claude/worktrees/mapanything-eval`
(branch `worktree-mapanything-eval-real`, created directly from
`apple-tree-cameras` HEAD `d4bd146` via `git worktree add` — NOT the default
EnterWorktree base, which branches from `origin/master` fresh and would have
missed Phase 7's committed `yogesh_dev/phase7/` deliverables including the
real WAI dataset). All new files confined to `yogesh_dev/mapanything_eval/`.

## What MapAnything actually is

Real repo: https://github.com/facebookresearch/map-anything (Meta/FAIR,
Nikhil Keetha et al., 2025). "Universal Feed-Forward Metric 3D
Reconstruction" — a single transformer model that takes 1-N images (any
subset optionally also given camera intrinsics, camera poses, and/or metric
depth) and predicts, for every view, a dense pointmap / depth + camera pose
+ intrinsics, feed-forward (no per-scene optimization), in real metric
scale when enough conditioning is provided. Confirmed via WebSearch +
WebFetch of the real README/HF model card (not guessed):
- Pretrained weights on HuggingFace Hub, safetensors format, NOT gated
  behind login (verified against `facebook/map-anything`,
  `facebook/map-anything-apache`, `-v1` variants) — no auth blocker.
- pip package name is `mapanything`, installed from source
  (`pip install -e .` inside a git clone of the repo), not a plain PyPI
  package name to guess at.

## Installation (real commands run)

```
cd yogesh_dev/mapanything_eval
git clone https://github.com/facebookresearch/map-anything.git
conda create -n mapanything python=3.12 -y
conda activate mapanything
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
cd map-anything && pip install -e .
```

Chose torch 2.7.0+cu128 specifically (not latest) because it's the exact
version already verified working with this machine's RTX 5090
(sm_120/compute capability 12.0) in the pre-existing `gsplat` conda env
(`torch.cuda.get_device_capability(0) == (12, 0)`, confirmed via
`/home/yogesh/anaconda3/envs/gsplat/bin/python`, not modified). New
dedicated env, nothing installed into `helios`/`gsplat`/`nerf`/`yolo`.

`pip install -e .` pulled in `torchaudio==2.11.0` as a transitive dep
(unpinned in `mapanything`'s `pyproject.toml`), which was ABI-incompatible
with cu128 (`libcudart.so.13: cannot open shared object file`) — but
`mapanything` never imports `torchaudio` directly (confirmed:
`from mapanything.models import MapAnything` succeeds regardless), so this
wasn't a real blocker for inference. Fixed for cleanliness anyway:
`pip install torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128`.

Verified after install: `torch 2.7.0+cu128`, `torch.cuda.is_available()==True`
on the RTX 5090.

## Smoke test (real weights download, real inference)

```
export HF_HOME=yogesh_dev/mapanything_eval/hf_cache
python -c "from mapanything.models import MapAnything; ...; model = MapAnything.from_pretrained('facebook/map-anything').to('cuda'); ..."
```

- No HuggingFace auth wall: `facebook/map-anything` (CC-BY-NC-4.0, 1B params,
  safetensors) downloaded unauthenticated, confirming the task's stated
  "stop cleanly if gated" condition never triggers here.
- First load also torch-hub-downloads a DINOv2 ViT-g/14 backbone
  (`facebookresearch/dinov2` zipball from GitHub) -- real, ~159s on this
  machine's connection the first time, cached after
  (`~/.cache/torch/hub/facebookresearch_dinov2_main`, outside the repo, not
  touched/committed).
- Real inference on 2 real WAI-scene images (`tree0_orbit16` frames 0,1):
  succeeded end to end, 0.445s, produced real `pts3d` (392x518x3 world
  points), `conf`, `mask`, etc. -- exact key set documented in
  `MapAnything.infer`'s docstring
  (`map-anything/mapanything/models/mapanything/model.py`).

**No blockers hit.** Everything in the "if gated, stop" escape hatch turned
out not to apply.

## WAI loader (`wai_loader.py`)

Reads Phase 7's real `scene_meta.json` + per-frame JPEG/EXR directly (real
`OpenEXR.File(path).channels()["Z"].pixels`, same convention Phase 7's own
`render_utils.py:_read_depth_exr` used) and builds MapAnything's real
per-view input schema (`img`, and for the "full" conditioning level also
`intrinsics`, `depth_z`, `camera_poses`, `is_metric_scale=True`) --
confirmed against `MapAnything.infer`'s real docstring and
`mapanything/utils/image.py:preprocess_inputs`, not guessed. WAI's
`transform_matrix` (camera-to-world, OpenCV convention) maps directly onto
MapAnything's `camera_poses` input (also "(4,4) OpenCV cam2world" per its
own docs) -- no convention conversion needed, verified in-repo by grepping
`mapanything/datasets/wai/scannetpp.py` for how MapAnything's OWN training
pipeline treats WAI-format scenes.

Invalid/sky depth pixels (Helios plane-depth sentinel is exactly -1.0,
Phase 0/7 convention) are zeroed before handing to MapAnything, matching
this repo's OWN convention for missing depth (`scannetpp.py`:
`np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)` -- 0 is this
codebase's real "no measurement" sentinel, not a guess).

Two conditioning levels implemented, matching two of Phase 7 D1's four
(images-only = A; full = C+D merged since MapAnything takes intrinsics +
extrinsics + depth simultaneously rather than as separate ablation arms
the way the classical proxy needed to structure them):
- `images_only`: only `img` per view.
- `full`: `img` + real `intrinsics` + real `camera_poses` (extrinsics) +
  real `depth_z`, `is_metric_scale=True` -- the fullest available
  conditioning.

## Real inference runs (`run_inference.py`)

All 4 real runs (2 scenes x 2 conditioning levels) completed on the RTX
5090, `facebook/map-anything`, `memory_efficient_inference=True`,
`use_amp=True` (bf16), no errors, no OOM:

| scene | conditioning | n_views | infer_s (wall clock, model already loaded) | s/view |
|---|---|---|---|---|
| tree0_orbit16 | images_only | 16 | 1.927 | 0.1204 |
| tree0_orbit16 | full | 16 | 1.996 | 0.1247 |
| tree_t76_mvs_rig | images_only | 42 | 5.064 | 0.1206 |
| tree_t76_mvs_rig | full | 42 | 5.413 | 0.1289 |

Model weight load (`MapAnything.from_pretrained`, warm HF+torch-hub cache):
~11.3s, one-time, amortized across all 4 runs in this process (real number,
not hidden) -- reported separately from inference latency since Phase 7's
proxy numbers also don't include one-time setup cost in their reported
`rmse_mm` figures, so this keeps the "latency" comparison apples-to-apples
(pure per-call inference time). Raw predictions (`pts3d`, `depth_z`,
`intrinsics`, `camera_poses`, `mask`, `conf` per view) saved to
`output/<scene>/<conditioning>/predictions.npz` + `run_meta.json`
(includes real ground-truth poses/intrinsics inline for downstream
comparison scripts). Merged, masked, real-world-frame point clouds also
exported per scene/conditioning to `world_points_merged.npy`/`.ply`
(`export_pointclouds.py`) -- viewable in MeshLab/CloudCompare/open3d.

## A real, load-bearing discovery: MapAnything's world frame is anchored to camera 0, NOT the input world frame

First pass at comparing predicted vs real camera poses gave nonsense (6.5m
RMSE, ~100deg rotation error) even for the "full" conditioning run where
REAL extrinsics were given as input for every view. Root cause, confirmed
by inspecting raw output: `pred["camera_poses"][0]` comes back ~identity
(translation ~1e-3, rotation ~1e-4 off) REGARDLESS of what absolute pose
was given for view 0 as conditioning -- MapAnything always defines its
output world frame as view 0's own camera frame; given `camera_poses`
conditioning fixes the RELATIVE geometry between views, not the caller's
absolute frame. Fix: re-reference both predicted and real ground-truth
poses to camera 0 (`inv(pose[0]) @ pose[i]`) before any comparison. After
the fix, "full" conditioning gives sane, real numbers (67-88mm camera
center RMSE, <1.2deg rotation error) -- see below. This single discovery
is the reason `compare_pose_accuracy.py`, `compare_landmark_rmse.py`, and
`compare_voxel_reconstruction.py` all explicitly re-reference frames before
comparing; skipping this step silently produces garbage numbers that look
like a broken model rather than a frame-convention mismatch.

Also observed: `metric_scaling_factor` in the output is NOT 1.0 even under
"full" (real extrinsics + real metric depth given) conditioning (e.g.
3.94x on `tree0_orbit16`/full) -- MapAnything applies its own internal
learned metric-scale-adaptor correction on top of whatever conditioning is
given, baked into the returned `pts3d`/`camera_poses` already (confirmed
by reading `mapanything/models/mapanything/model.py`'s `scale_adaptor` /
`scale_final_output`, not guessed). This is a real architectural detail,
not a bug -- no extra correction needed downstream since it's already
applied to the tensors we read.

## Real camera-pose accuracy (`compare_pose_accuracy.py`, `output/pose_accuracy_report.json`)

| scene | conditioning | metric | value |
|---|---|---|---|
| tree0_orbit16 | full | camera-center RMSE | 67.2 mm |
| tree0_orbit16 | full | rotation error (mean / max) | 0.46 / 0.81 deg |
| tree0_orbit16 | images_only | camera-center RMSE (after Umeyama align) | 4640.9 mm |
| tree0_orbit16 | images_only | recovered scale factor | 8.75 |
| tree_t76_mvs_rig | full | camera-center RMSE | 87.8 mm |
| tree_t76_mvs_rig | full | rotation error (mean / max) | 0.46 / 1.14 deg |
| tree_t76_mvs_rig | images_only | camera-center RMSE (after Umeyama align) | 4554.6 mm |
| tree_t76_mvs_rig | images_only | recovered scale factor | 2.75 |

Real finding: given real extrinsics as conditioning, MapAnything's own
re-estimated camera poses stay very close (<9cm, <1.2deg) to what was fed
in -- it doesn't blindly parrot the input, it re-estimates and lands close.
Without any pose conditioning, camera-trajectory recovery is poor at this
scene's scale/complexity (~4.5m RMSE after best-fit similarity alignment,
recovered scale wrong by 2.75x-8.75x) -- notably worse than Phase 7's own
classical proxy found for ITS OWN much smaller/simpler 8-view arc rig in
T7.2 condition B (`t72_pose_conditioning_ablation.json`:
`recovered_scale_factor=10.2`, comparable order of magnitude actually,
though T7.2 evaluated 2-view essential-matrix recovery on a wide-baseline
pair, not a full 16/42-view joint estimate -- not a strict apples-to-apples
rig, flagged as such).

## Real landmark-recovery RMSE vs Phase 7's T7.2 conditioning axis (`compare_landmark_rmse.py`, `output/landmark_rmse_report.json`)

Real branch tube-segment midpoints (n=1177) + real fruit centroids (n=33),
same tree/ground-truth files T7.6/T7.7 used, `tree_t76_mvs_rig`'s real
42-view WAI scene:

| conditioning | MapAnything RMSE (mm) | Phase 7 T7.2 proxy RMSE (mm), same axis |
|---|---|---|
| images_only | 13872.0 | 74.4 (condition A, images only) |
| full (+intrinsics+extrinsics+depth) | 398.8 | 4.6 (condition C, +intrinsics+extrinsics) |

Real, honest finding, NOT flattering to MapAnything on this axis: even with
full conditioning, MapAnything's dense per-pixel landmark recovery is
~86x worse (398.8mm vs 4.6mm) than classical analytic DLT triangulation
of the SAME known 3D points from the SAME known poses. This is not
necessarily a fair fight (T7.2's condition C is sparse triangulation of
EXACTLY the queried point from N=8 known-correspondence views -- the best
case for classical geometry; MapAnything is dense monocular-ish per-pixel
depth regression sampled at a projected pixel, inherently noisier per
sample), but the magnitude of the gap is real and worth reporting plainly
rather than rounded away. Images-only conditioning on this scene is far
worse still (13.9m RMSE) -- an order of magnitude worse than Phase 7's own
image-only proxy on its simpler rig, showing this specific dense, thin,
self-similar-branch, 42-view/3-ring canopy capture is a hard case for
MapAnything's unconditioned reconstruction, not just "somewhat harder."
Full caveats on rig/tree/paradigm mismatch are in the JSON output's
`vs_phase7_t72.caveat` field, not hidden.

## Real voxel-grid reconstruction vs Phase 7's T7.6 classical baseline / T7.7 metric-scale integrity (`compare_voxel_reconstruction.py`, `output/voxel_reconstruction_report.json`)

Same real tree, same real ground truth, same real grid definition
(bmin/voxel_size loaded from T7.6's own persisted `.npz` files), same
recall functions (`thin_structure_recall._theoretical_recall` /
`_empirical_recall`, imported directly, not re-derived) -- MapAnything's
"full" conditioning point cloud fused into that exact grid via plain
point-voxelization (no free-space carving, see caveat below):

| metric (fine 5mm grid) | MapAnything (full) | Phase 7 T7.6 classical (exact poses) |
|---|---|---|
| n_occupied_voxels | 755,220 | 32,941 |
| branch recall <5mm | 100.0% | 72.7% |
| branch recall 5-10mm | 85.3% | 76.0% |
| branch recall 10-20mm | 87.7% | 88.1% |
| branch recall >20mm | 95.9% | 67.8% |
| fruit diameter mean abs rel err | 29.9% | 22.5% (T7.7) |
| internode length rel err | 5.4% | 11.2% (T7.7) |

**Important caveat, stated plainly, not hidden**: MapAnything's fused cloud
occupies ~23x more voxels than the classical baseline at the same
resolution (755K vs 33K) because this script does plain "any predicted
point marks its voxel occupied" fusion, NOT T7.6's real log-odds
probabilistic fusion with free-space carving. A much denser occupied grid
makes the recall check's "is there an occupied voxel within 1 voxel of
this landmark" easier to satisfy somewhat independent of reconstruction
QUALITY -- so the branch-recall win above is real (these are MapAnything's
actual predicted points, not fabricated) but is not a clean apples-to-apples
fusion-method comparison; some of the recall gap is attributable to fusion
density, not reconstruction accuracy alone. The fruit-diameter and
internode-length numbers are less sensitive to this (fruit diameter uses a
tight, GT-centered search radius; internode length snaps to nearest
occupied voxel, not "any occupied voxel nearby"), so the fruit-diameter
LOSS (MapAnything worse, 29.9% vs 22.5%) and internode-length WIN
(MapAnything better, 5.4% vs 11.2%) are more directly trustworthy as
reconstruction-quality comparisons than the branch-recall numbers.

Images-only conditioning: complete collapse at this task -- 0 occupied
voxels landed inside the real tree's bounding grid at all (consistent with
the 13.9m landmark RMSE and ~4.5m camera-trajectory RMSE found above; the
Umeyama-aligned reconstruction scatter is simply too large relative to the
tree's ~1.3m x 1.8m x 2.9m real bounding box for any point to land inside
it -- verified directly: aligned point-cloud centroid ends up around
(5.5, 11.9, 5.0), nowhere near the real bbox -- this is a real finding,
not a bounds-check bug). Canopy volume, fruit diameter, and internode
length are all unmeasurable (0/null) for images_only -- reported as such,
not papered over.

## Summary

Real, working, end-to-end MapAnything installation and inference against
Phase 7's real WAI dataset, no shortcuts, no proxy standing in for the real
model this time -- the exact gap Phase 7 flagged as unfillable without
installing a real foundation model:

1. **No blockers.** No HF gating, no incompatible-dependency dead end
   (the one dependency wrinkle, `torchaudio`, wasn't even load-bearing for
   inference). RTX 5090 + torch 2.7.0+cu128 (same combo already proven to
   work in this machine's `gsplat` env) ran real inference with no CUDA
   issues.
2. **Latency**: ~0.12-0.13s/view real wall-clock inference (16 or 42 views,
   `memory_efficient_inference=True`, bf16 AMP), ~11.3s one-time model
   load. Real numbers, not estimated -- this was the task's original
   motivation and Phase 7 explicitly could not measure it.
3. **Full conditioning (+intrinsics+extrinsics+depth) is dramatically
   better than images-only** for MapAnything on this data, same
   directional finding as Phase 7's classical-proxy T7.2 conditioning
   axis -- but the classical proxy's own numbers are substantially
   better in absolute terms on every metric checked (landmark RMSE 4.6mm
   vs MapAnything's 398.8mm under full conditioning; both collapse badly
   without conditioning, but MapAnything collapses harder here: 13.9m vs
   74mm landmark RMSE, 0 vs some nonzero occupied-grid overlap).
4. **Real bright spots for MapAnything under full conditioning**: better
   thin-structure branch recall than T7.6's classical fusion at 3 of 4
   diameter classes (with the important density caveat above) and better
   internode-length accuracy (5.4% vs 11.2% relative error) -- it is not a
   uniform loss against the classical baseline.
5. **Real weak spot**: fruit-diameter reconstruction accuracy is worse
   than the classical baseline (29.9% vs 22.5% mean absolute relative
   error) even under full conditioning.
6. **Images-only conditioning is a real, severe failure mode** on this
   specific data (dense 16-42 view canopy captures of thin, self-similar
   branch structure) -- worth flagging for anyone considering MapAnything
   for uncalibrated/unposed capture of vegetation-like scenes: this is not
   the regime it was strongest in during this test.

All numbers above are real MapAnything outputs on real Helios-rendered
data, real Phase 7 ground truth, and real Phase 7 proxy results -- nothing
in this section is estimated, extrapolated, or a placeholder.
