# Foundation-model evaluation log — DA3, Pi3X, VGGT, MoGe-2, PromptDA

Follow-on to the MapAnything evaluation (`yogesh_dev/mapanything_eval/`, worktree
branch `worktree-mapanything-eval-real`, also copied to `/home/yogesh/PyHelios/mapanything/`).
Same real data, same real ground truth, same scoring code paths, so every number
below is directly comparable to MapAnything's already-published real numbers.

Working directory: isolated git worktree `.claude/worktrees/foundation-models-eval`
(branch `worktree-foundation-models-eval`, created with `git worktree add ... HEAD`
directly from `apple-tree-cameras` HEAD `d4bd146` — deliberately NOT the default
`EnterWorktree` base, which branches from `origin/master` and would have missed
Phase 7's committed `yogesh_dev/phase7/` deliverables including the real WAI
dataset, exactly as the MapAnything eval found). All new files confined to
`yogesh_dev/foundation_models_eval/`.

Machine: RTX 5090 (32 GB, compute capability 12.0 / sm_120, driver 580.173.02).
**This is load-bearing for every install below**: sm_120 needs CUDA 12.8+ kernels,
and three of the five repos pin a torch version that predates Blackwell support
(Pi3 pins `torch==2.5.1`, VGGT pins `torch==2.3.1`, PromptDA pins `torch==2.0.1`).
Every environment here therefore installs `torch==2.7.0+cu128` instead of the
pinned version — the same combination already proven working on this machine in
the pre-existing `gsplat` env and reused by the MapAnything eval. Each deviation
is recorded per-model below rather than silently applied.

## Real data used (unchanged from the MapAnything eval — that is the point)

- `yogesh_dev/phase7/output/wai_dataset/helios_apple_tree/tree0_orbit16` — 16 real views
- `yogesh_dev/phase7/output/wai_dataset/helios_apple_tree/tree_t76_mvs_rig` — 42 real views
- Real RGB (480x360 JPEG), real EXR plane depth (`-1.0` = sky/no-hit sentinel),
  real per-frame `transform_matrix` (cam2world) and `world_to_camera_matrix` (w2c),
  shared real intrinsics (fl=434.558 px, cx/cy=240/180).
- Real ground truth: `t76_branch_segments.json` (1177 branch tube-segment
  midpoints), `t76_ground_truth_scale.json` (33 fruit centroids/diameters,
  canopy bbox, mean internode length), `t76_grid_coarse.npz` / `t76_grid_fine.npz`
  (Phase 7 T7.6's own classical-fusion occupancy grids and grid definitions).

**A measured property of this data that turned out to explain a lot of the
results below**: only **6.04%** of pixels in these scenes are a real ray hit.
The other ~94% is open sky behind a thin canopy. Measured directly, both scenes
agree (0.0604 / 0.0605). Valid depth spans 3.78–6.73 m, median ~5.1 m, and the
tree's own depth extent is only ~1.2 m front-to-back. So these are thin,
sparse, low-texture targets at medium range against a featureless background —
close to a worst case for monocular scale inference, and the direct cause of
PromptDA's headline failure mode (see below).

## Shared harness (written once, used by every model)

| file | purpose |
|---|---|
| `wai_loader.py` | Reads the real WAI scenes. Deliberately a near-copy of the MapAnything eval's real reading logic (same `OpenEXR.File(path).channels()["Z"].pixels`, same `-1.0` sky sentinel) so every model sees byte-identical inputs. |
| `prediction_bundle.py` | One on-disk schema for the 3D-reconstruction family, byte-compatible with what the MapAnything eval already wrote (`pts3d__i`, `mask__i`, `intrinsics__i`, `camera_poses__i` + `run_meta.json`). |
| `depth_bundle.py` | Separate schema + metrics for the depth-focused family. |
| `frame_policy.py` | Decides empirically how each model's output frame relates to the real world frame. See "The load-bearing discovery" below. |
| `compare_pose_accuracy.py`, `compare_landmark_rmse.py`, `compare_voxel_reconstruction.py` | Ports of the MapAnything eval's three scoring scripts, generalized over `--models`. Import Phase 7's own `mv_geometry.umeyama_alignment` / `rmse_rigid` and `thin_structure_recall._theoretical_recall` / `_empirical_recall` directly — not re-derived. |
| `compare_depth.py` | Per-view depth metrics + two real reference floors (Phase 1's sensor-noise model, and naive upsampling). |
| `make_comparison_tables.py` | Emits `output/COMPARISON_TABLES.md` straight from the report JSONs, including MapAnything's column read live from its own `output/*.json`. **No number in this log is hand-transcribed.** |

Because all five models write one of two shared schemas, the *same* scoring code
scores every model. A difference between two rows is a real model difference,
not a difference in how each was measured.

---

## The load-bearing discovery: three models, three different frame conventions

The MapAnything eval learned the hard way that MapAnything anchors its output
world frame to view 0's own camera frame, and that skipping the re-referencing
step silently produces garbage. Running four models through the same harness
surfaced that **there is no single convention to assume** — each behaves
differently when handed real camera poses as conditioning:

| model | what it does with real extrinsics given as conditioning | verified how |
|---|---|---|
| **MapAnything** | Anchors output to view 0's camera frame, but preserves the input's metric scale. Re-estimates poses and lands 67–88 mm / <1.2° from them. | mapanything_eval's own finding |
| **DA3** | Returns the caller's input extrinsics **verbatim** and rescales depth to match. | `api.py:_align_to_input_extrinsics_intrinsics` assigns `prediction.extrinsics = extrinsics[..., :3, :]`; measured `max abs(pred_c2w - gt_c2w) = 4.8e-07` |
| **Pi3X** | Keeps its **own** canonical frame *and* its own scale, even when given real poses AND real metric depth. Conditioning constrains relative geometry, not the caller's absolute gauge. | measured recovered scale 1.99–2.93× off, absolute translations nowhere near the inputs' |
| **VGGT** | N/A — no pose conditioning input exists. | architectural |

`frame_policy.py` therefore decides per run, empirically: fit a Umeyama
similarity from predicted camera centres onto real ones (after re-referencing
both to camera 0); if the recovered scale is within 2% of 1.0 **and** the
direct residual is under 50 mm, the model honoured the caller's gauge and the
identity transform is used; otherwise the fitted similarity is applied and the
recovered scale is reported explicitly as part of the finding. Nothing observed
landed between those cases (DA3 measured 4.8e-7 relative; Pi3X measured 99–193%
off), so the tolerance is not a knife-edge choice. **Both** the direct and the
aligned residual are always written out.

**Consequence for reading the tables**: DA3's `full` camera-pose RMSE of exactly
0.0 mm is *not* a model achievement — it is an API pass-through. DA3's pose
accuracy under conditioning is simply **not measurable** through its public
`inference()` API. Its reconstruction quality under conditioning still is, and
that is what the landmark/voxel tables measure.

---

## Per-model: what it is, real install, real blockers

### 1. Depth Anything 3 (DA3) — DONE, no blockers

Real repo: https://github.com/ByteDance-Seed/Depth-Anything-3 (ByteDance Seed).
Code Apache-2.0 (verified: repo `LICENSE` is the Apache 2.0 text, `pyproject.toml`
declares `license = { text = "Apache-2.0" }`). A single "any-view" transformer
predicting depth + camera pose + intrinsics from 1–N images, optionally
**conditioned on known camera poses** — the property the design doc cares about.

**Weight licences verified per checkpoint from the repo's own model-zoo table
before downloading anything** (they differ, and the split is not what a guess
would produce):

| checkpoint | params | pose cond. | licence |
|---|---|---|---|
| `DA3-BASE` | 0.12B | yes | **Apache 2.0** |
| `DA3-SMALL` | 0.08B | yes | **Apache 2.0** |
| `DA3-LARGE-1.1` | 0.35B | yes | CC BY-NC 4.0 |
| `DA3-GIANT-1.1` | 1.15B | yes | CC BY-NC 4.0 |
| `DA3NESTED-GIANT-LARGE-1.1` | 1.40B | yes | CC BY-NC 4.0 |
| `DA3METRIC-LARGE` | 0.35B | **no** | Apache 2.0 |
| `DA3MONO-LARGE` | 0.35B | **no** | Apache 2.0 |

A correction worth flagging against the task's framing: `DA3METRIC-LARGE` and
`DA3MONO-LARGE` are Apache-2.0 but are **monocular-only** — the HF card for
DA3METRIC-LARGE states "specialized for metric depth estimation in monocular
settings", and the repo's table shows no Pose-Est./Pose-Cond. capability for
either. So the largest *pose-conditionable* Apache-2.0 checkpoint is only
`DA3-BASE` (0.12B). Both were run: **DA3-BASE** as the licence-clean primary and
**DA3-LARGE-1.1** (CC BY-NC 4.0, research-only, flagged) as the quality
representative.

```bash
git clone --recursive https://github.com/ByteDance-Seed/Depth-Anything-3.git
conda create -n fm_da3 python=3.12 -y && conda activate fm_da3
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
cd Depth-Anything-3 && pip install -e .
pip install OpenEXR      # needed by wai_loader, not by DA3 itself
```
Installed clean: torch 2.7.0+cu128, numpy 1.26.4 (DA3 pins `numpy<2`), xformers
0.0.30. No HF auth wall on any checkpoint used. `DA3-BASE` load 16.9 s cold /
0.9 s warm; `DA3-LARGE-1.1` load 59.3 s cold.

Conditioning levels run (what `inference()` architecturally accepts — verified
by reading `src/depth_anything_3/api.py`, not guessed):
- `images_only` — camera-decoder pose head
- `images_only_ray` — `use_ray_pose=True`, the ray head the design doc asks about
- `full` — images + real intrinsics + real extrinsics (w2c)

**DA3 has no depth-conditioning input.** `inference()` takes only `extrinsics`
and `intrinsics`. So DA3's "full" is *weaker* conditioning than MapAnything's
"full" (which also ingested real metric `depth_z`). That is an architectural
difference, not a skipped arm, and it makes DA3's results below more impressive
rather than less.

Validity mask: DA3's own default point-export rule, copied exactly from
`utils/export/glb.py` (`conf_thresh=1.05` clipped into
`[percentile(conf,40), percentile(conf,90)]`, then `conf > threshold`), plus its
own predicted sky mask. Never ground-truth depth.

### 2. Pi3X — DONE, no blockers

Real repo: https://github.com/yyfz/Pi3 (ICLR 2026). Pi3X is the December 2025
upgrade of π³, and it *was* cleanly findable and reproducible — no fallback to
base π³ needed. Code **BSD-3-Clause**; **weights `yyfz233/Pi3X` are CC BY-NC 4.0**
(the repo's own table: "Strictly Non-Commercial"). Fine for this research
evaluation; **it would not be usable commercially**, flagged here rather than
buried.

```bash
git clone https://github.com/yyfz/Pi3.git
conda create -n fm_pi3 python=3.12 -y && conda activate fm_pi3
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy==1.26.4 pillow opencv-python plyfile huggingface_hub safetensors OpenEXR einops
```
Deviation: `requirements.txt` pins `torch==2.5.1`, which has no sm_120 kernels;
installed 2.7.0+cu128 instead and everything worked. Warning at runtime
`cannot find cuda-compiled version of RoPE2D, using a slow pytorch version` —
cosmetic, affects speed only, inference succeeded. Model load 156.7 s cold / ~7 s warm.

Pi3X is the **only** model here that accepts all three modalities MapAnything did.
Levels run: `images_only`, `pose_intrinsics` (+ real intrinsics + real cam2world
poses), `full` (+ real metric depth as well).

**A real trap avoided, worth recording**: Pi3's own
`pi3/utils/basic.py:load_multimodal_data` discovers images with
`sorted(os.listdir(...))`. Phase 7's frames are named `cam0_frame_00000.jpeg`,
`cam1_...`, `cam10_...`, so that sort yields cam0, cam10, cam11, … — a *different*
order from `scene_meta.json`'s frame list, which the pose/depth/intrinsics
condition arrays are indexed by. Calling it directly would have silently paired
each image with another frame's pose and produced plausible-looking nonsense.
`run_pi3x.py:prepare_pi3_inputs` reproduces its resize maths exactly
(PIXEL_LIMIT=255000, LANCZOS, both dims forced to multiples of 14 by the same
shrink loop, intrinsics scaled by scale_x/scale_y, depth resized INTER_NEAREST)
but drives it from the explicit scene_meta order.

Validity mask: Pi3's own `example_mm.py` rule verbatim
(`sigmoid(conf) > 0.1` AND not `depth_normal_edge(..., rtol=0.03)`). This is an
aggressive edge filter and keeps only 5.5–8.5% of pixels on this canopy — noted
because it materially limits Pi3X's fused point count.

### 3. VGGT — DONE, with a real access blocker on the licence-clean checkpoint

Real repo: https://github.com/facebookresearch/vggt. Included per the design doc
as the **uncalibrated comparison anchor**: `model.aggregator(images)` takes images
and nothing else, and the camera head *predicts* extrinsics/intrinsics rather
than accepting them, so there is exactly one conditioning level and that is
architectural, not a skipped arm.

**Real blocker, recorded rather than glossed over**: the licence-clean
`facebook/VGGT-1B-Commercial` checkpoint was tried first and is **gated**:

```
huggingface_hub.errors.GatedRepoError: 401 Client Error.
Cannot access gated repo for url https://huggingface.co/facebook/VGGT-1B-Commercial/resolve/main/model.safetensors.
Access to model facebook/VGGT-1B-Commercial is restricted. You must have access to it and be authenticated to access it.
```

No unauthenticated download is possible. Fell back to `facebook/VGGT-1B`, which
is **not** gated but **is** non-commercial (VGGT's own README: "only the newly
released checkpoint VGGT-1B-Commercial is licensed for commercial usage — the
original checkpoint remains non-commercial"). Same footing as Pi3X's NC weights
for research purposes; **any commercial use would need the gated checkpoint**.

```bash
git clone https://github.com/facebookresearch/vggt.git
conda create -n fm_vggt python=3.12 -y && conda activate fm_vggt
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy==1.26.4 pillow huggingface_hub einops safetensors opencv-python OpenEXR
```
Deviation: `requirements.txt` pins `torch==2.3.1` (no sm_120); used 2.7.0+cu128.
Model load 154.0 s cold / 7.6 s warm.

3D points built with VGGT's own `unproject_depth_map_to_point_map(depth,
extrinsic, intrinsic)`, which its README explicitly recommends over the point
head. Extrinsics are w2c OpenCV per its README, inverted to cam2world here.

**A confound found and controlled for**: VGGT's own scripted default confidence
threshold (`demo_colmap.py --conf_thres_value`, default 5.0) keeps only **0.4%**
of pixels on `tree0_orbit16` and 5.2% on `tree_t76_mvs_rig` — far stricter than
any other model here. An RMSE computed over only the model's easiest pixels
would flatter it. So a second arm `images_only_laxconf` (`conf > 1.0`, ~13–14%
of pixels) was run to separate accuracy from selectivity. Both are reported.

### 4. MoGe-2 — DONE, no blockers

Real repo: https://github.com/microsoft/MoGe. Code **MIT** (verified: repo
`LICENSE` is the MIT text), except vendored DINOv2 under Apache-2.0. Checkpoint
`Ruicheng/moge-2-vitl-normal`, downloaded unauthenticated, no gating.

```bash
git clone https://github.com/microsoft/MoGe.git
conda create -n fm_depth python=3.11 -y && conda activate fm_depth
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy==1.26.4 opencv-python scipy matplotlib trimesh pillow huggingface_hub click OpenEXR einops imageio tqdm safetensors
pip install "git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183"
cd MoGe && pip install -e .
```

Monocular metric geometry: one image in, one metric depth/point map out. Its
`infer()` signature accepts exactly one optional geometric input, `fov_x`, so
the two levels run are `images_only` (FoV inferred) and `known_fov` (real
horizontal FoV computed from Phase 7's real intrinsics). **There is no pose or
depth conditioning input**, so there is no "full" arm — architectural.
Evaluated against real per-view EXR depth, not through the reconstruction
tables, because per-view metric depth is what it actually produces.

### 5. PromptDA — DONE, no blockers (after one real diagnostic)

Real repo: https://github.com/DepthAnything/PromptDA (CVPR 2025). Code
**Apache-2.0** (verified: repo `LICENSE` is the Apache 2.0 text). Checkpoint
`depth-anything/prompt-depth-anything-vitl`, downloaded unauthenticated, no
gating. Installed into the same `fm_depth` env; `pip install -e .` needed one
retry (first attempt registered the dist but left the package unimportable;
a plain re-run fixed it).

Deviation: `requirements.txt` pins `torch==2.0.1` / `xformers==0.0.22`; used
torch 2.7.0+cu128 and did not install xformers — inference succeeded.

Takes RGB **plus a low-resolution noisy metric depth "prompt"** (designed around
iPhone ARKit LiDAR) and outputs dense metric depth. Per the task, the prompt is
built from the real GT depth in the order a real sensor would produce it:

```
real dense GT depth (metres, -1.0 = sky)
  -> Phase 1's REAL RGB-D noise model, at full resolution
     (yogesh_dev/phase1/noise_model.py:apply_rgbd_noise_model, imported and
      called directly, seed 0 — mixed/flying pixels at genuine depth
      discontinuities + range-dependent sigma(z) = 0.0015 * z^2)
  -> low-resolution readout (INTER_NEAREST, 4x — matching the 3.94x ratio
     PromptDA's own example ships with: 1008x756 image, 256x192 ARKit depth)
```

Noise **first**, then subsample, because that is the physical order — doing it
the other way would filter the mixed-pixel artifact away with the subsampling
and quietly make the task easier than reality. Ground truth for scoring is
always the real, clean, dense EXR depth; the noisy/sparse version is only ever
an input.

Resolution: PromptDA's ViT has patch size 14 and its own loader floors each
dimension to a multiple of 14. 480x360 is neither, so images are resized to
476x350 and scored against GT resampled to that grid with INTER_NEAREST
(nearest everywhere for depth, so no value is ever invented between two
surfaces straddling an occlusion boundary).

---

## Running comparison tables

Full machine-generated tables: **`output/COMPARISON_TABLES.md`** (regenerate with
`python make_comparison_tables.py`). MapAnything's rows there are read live from
its own `output/*.json`. Key extracts follow.

### Landmark-recovery RMSE — the headline axis (`tree_t76_mvs_rig`, 1177 real branch midpoints + 33 real fruit centroids)

| model | conditioning | given real poses? | landmark RMSE (mm) | median (mm) | n landmark-view obs |
|---|---|---|---|---|---|
| **Phase 7 classical proxy (T7.2 cond. C)** | +intrinsics+extrinsics | yes | **4.6** | — | — |
| **Phase 7 classical proxy (T7.2 cond. A)** | images only | no | **74.4** | — | — |
| da3_base | full (+K+extrinsics) | yes | **377.2** | 232.6 | 50734 |
| **vggt_1b** | **images_only** | **no** | **381.0** | 245.2 | 816 |
| da3_large11 | full (+K+extrinsics) | yes | 398.4 | 233.7 | 50820 |
| mapanything | full (+K+extrinsics+**depth**) | yes | 398.8 | 232.7 | 47882 |
| vggt_1b | images_only_laxconf | no | 451.6 | 253.8 | 1159 |
| pi3x | pose_intrinsics (+K+extrinsics) | yes | 1071.8 | 822.4 | 2289 |
| da3_large11 | images_only_ray | no | 2112.9 | 2133.8 | 879 |
| da3_large11 | images_only | no | 2849.6 | 2134.1 | 929 |
| pi3x | full (+K+extrinsics+**depth**) | yes | 2990.8 | 2971.4 | 1395 |
| da3_base | images_only | no | 4775.7 | 4233.7 | 2637 |
| pi3x | images_only | no | 5266.9 | 4576.5 | 1294 |
| mapanything | images_only | no | 13872.0 | 13790.2 | 36261 |
| da3_base | images_only_ray | no | n/a (0 obs) | — | 0 |

`n landmark-view obs` matters and is not decoration: a model whose own validity
mask keeps very few pixels is scored on far fewer, easier samples. DA3 `full`
contributes ~50k observations; VGGT contributes 816. The `images_only_laxconf`
row exists precisely to test whether VGGT's lead survives loosening its mask —
it does (381.0 → 451.6 mm while observations grow 42%), so VGGT's accuracy is
real and not purely a selection effect, but it remains measured on ~2% as many
samples as DA3.

### Voxel-grid reconstruction vs Phase 7's classical baseline (`tree_t76_mvs_rig`, fine 5 mm grid)

| model | conditioning | n occupied voxels | recall <5mm | 5-10mm | 10-20mm | >20mm | fruit diam abs rel err | internode rel err |
|---|---|---|---|---|---|---|---|---|
| **Phase 7 T7.6/T7.7 classical** | exact poses, log-odds fusion | 32,941 | 0.727 | 0.760 | 0.881 | 0.678 | **0.2245** | 0.1124 |
| da3_base | full | 1,238,877 | **0.939** | 0.797 | 0.848 | 0.863 | 0.2988 | **0.0584** |
| da3_large11 | full | 1,765,992 | 0.848 | **0.861** | **0.930** | 0.740 | 0.2990 | 0.0656 |
| mapanything | full (+depth) | 755,220 | 1.000 | 0.853 | 0.877 | **0.959** | 0.2987 | 0.0536 |
| pi3x | pose_intrinsics | 849,545 | 0.758 | 0.573 | 0.713 | 0.541 | 0.2971 | 0.0566 |
| vggt_1b | images_only_laxconf | 988,137 | 0.515 | 0.523 | 0.611 | 0.637 | 0.2922 | 0.0372 |
| vggt_1b | images_only | 419,207 | 0.182 | 0.284 | 0.287 | 0.308 | 0.2846 | **-0.0052** |
| da3_large11 | images_only | 271,026 | 0.121 | 0.045 | 0.156 | 0.075 | 0.2905 | -0.0664 |
| pi3x | full | 35,912 | 0.061 | 0.011 | 0.037 | 0.014 | 0.2729 | -0.6888 |
| da3_base / pi3x | images_only | 0 | 0.000 | 0.000 | 0.000 | 0.000 | n/a | n/a |

**The same caveat the MapAnything eval flagged still applies and is not fixed
here**: Phase 7's classical grid came from real log-odds probabilistic depth
fusion *with free-space carving*; every model row above is plain point-cloud
voxelization of mask-valid predicted points — a simpler, weaker fusion step that
also produces a **25–50× denser** occupied grid. A denser grid makes "is there
an occupied voxel within one voxel of this landmark" easier to satisfy somewhat
independently of reconstruction quality, so the branch-recall wins are partly a
fusion-density artifact. **The fruit-diameter and internode-length columns are
much less sensitive to this** (tight GT-centred search radius; nearest-occupied
snap) and are the more trustworthy reconstruction-quality signals. On those:
every model is *worse* than classical on fruit diameter (0.27–0.30 vs 0.2245)
and *better* on internode length (0.037–0.066 vs 0.1124).

### Camera-pose accuracy

| model | scene | conditioning | RMSE direct (mm) | RMSE aligned (mm) | recovered scale | rot err aligned (deg) |
|---|---|---|---|---|---|---|
| mapanything | tree0_orbit16 | full | **67.2** | — | 1.000 | 0.46 |
| mapanything | tree_t76 | full | **87.8** | — | 1.000 | 0.46 |
| da3_* | both | full | 0.0 (API echo, not measurable) | — | 1.000 | 0.00 |
| pi3x | tree0_orbit16 | pose_intrinsics | 3917.9 | **277.9** | 2.049 | 2.32 |
| pi3x | tree0_orbit16 | full | 3614.4 | **257.2** | 1.985 | 1.96 |
| pi3x | tree_t76 | pose_intrinsics | 4217.8 | 499.0 | 2.166 | 3.10 |
| pi3x | tree_t76 | full | 5244.0 | 1321.8 | 2.930 | 8.06 |
| vggt_1b | tree0_orbit16 | images_only | 6189.4 | **741.8** | 5.615 | 6.53 |
| vggt_1b | tree_t76 | images_only | 6338.6 | 1557.1 | 5.160 | 6.45 |
| pi3x | tree0_orbit16 | images_only | 4550.2 | 1053.4 | 2.271 | 10.37 |
| da3_large11 | tree0_orbit16 | images_only | 7898.8 | 2507.9 | 6.004 | 26.56 |
| pi3x | tree_t76 | images_only | 6032.5 | 2969.9 | 3.270 | 51.85 |
| da3_large11 | tree_t76 | images_only | 6882.6 | 3173.0 | 6.640 | 48.33 |
| mapanything | both | images_only | — | 4554–4641 | 2.75 / 8.75 | — |
| da3_base | both | images_only | 7619–7888 | 3674–4532 | 6.9–16.0 | 76–92 |

**Every unconditioned model gets absolute scale badly wrong on this scene** —
recovered scale factors run 2.27× to 16.0×. Not one recovers metric scale from
images alone. That is the single most consistent finding across all four
reconstruction models and it directly supports the design doc's emphasis on
pose-conditioned modes for this domain.

### Real latency (RTX 5090, bf16 AMP, model already loaded)

| model | s/view, 16-view scene | s/view, 42-view scene | one-time load (cold) |
|---|---|---|---|
| da3_base | 0.0245–0.0486 | 0.0160–0.0217 | 16.9 s |
| da3_large11 | 0.0304–0.0538 | 0.0294–0.0374 | 59.3 s |
| vggt_1b | 0.0547–0.0574 | 0.0444–0.0446 | 154.0 s |
| pi3x | 0.0505–0.0884 | 0.0558–0.0770 | 156.7 s |
| **mapanything (reference)** | **0.1204–0.1247** | **0.1206–0.1289** | **11.3 s** |
| moge2_vitl (per-image, monocular) | 0.0449–0.0596 | 0.0448 | — |
| promptda_vitl (per-image) | 0.0318–0.0538 | 0.0319–0.0321 | — |

**Every model tested here is faster per view than MapAnything**, DA3-BASE by
5–7×. Pose-conditioned DA3 runs are consistently *faster* than unconditioned
ones (0.016 vs 0.018–0.022 s/view on the 42-view scene) — supplying poses
removes work rather than adding it.

### Per-view depth accuracy — depth-focused family (real EXR ground truth)

Scored on real ray-hit pixels only (the 6.04% that are actually tree).

| model | scene | conditioning | RMSE (mm) | MAE (mm) | AbsRel | δ1 | scale-aligned RMSE (mm) | median scale pred→gt |
|---|---|---|---|---|---|---|---|---|
| **Phase 1 real sensor-noise floor** | tree0_orbit16 | — | **98.3** | 67.4 | 0.0131 | 1.000 | 98.5 | 0.997 |
| **Phase 1 real sensor-noise floor** | tree_t76 | — | **93.9** | 62.6 | 0.0124 | 1.000 | 93.7 | 0.998 |
| naive 4× nearest upsampling of the same noisy prompt | tree0_orbit16 | — | 1451.0 | — | 0.1671 | — | — | — |
| naive 4× nearest upsampling of the same noisy prompt | tree_t76 | — | 1506.1 | — | 0.1756 | — | — | — |
| promptda_vitl | tree0_orbit16 | prompt_sparse_noisy **_bgfill** | **886.2** | 499.4 | 0.0970 | 0.893 | 842.5 | **0.983** |
| promptda_vitl | tree_t76 | prompt_sparse_noisy **_bgfill** | **900.6** | 503.5 | 0.0997 | 0.894 | 870.2 | **0.988** |
| promptda_vitl | tree0_orbit16 | prompt_sparse_noisy (sky→0) | 4897.8 | 4754.8 | 0.9209 | 0.051 | 132723 | 101.9 |
| promptda_vitl | tree0_orbit16 | prompt_sparse_clean (sky→0) | 4897.6 | 4755.3 | 0.9210 | 0.051 | 130918 | 100.2 |
| promptda_vitl | tree0_orbit16 | prompt_dense_noisy (sky→0) | 4836.8 | 4669.4 | 0.9041 | 0.062 | 117967 | 79.6 |
| promptda_vitl | tree_t76 | prompt_sparse_noisy (sky→0) | 4770.7 | 4616.0 | 0.9109 | 0.056 | 106730 | 78.9 |
| moge2_vitl | tree0_orbit16 | known_fov | 5750.8 | 5670.6 | 1.1083 | 0.000 | 561.7 | 0.471 |
| moge2_vitl | tree_t76 | known_fov | 5276.1 | 5120.5 | 1.0260 | 0.000 | 771.1 | 0.509 |
| moge2_vitl | tree_t76 | images_only | 12527.8 | 11798.4 | 2.3446 | 0.000 | 1521.7 | 0.331 |
| moge2_vitl | tree0_orbit16 | images_only | 14663.3 | 13959.8 | 2.7166 | 0.000 | 1234.2 | 0.267 |

Two real reference floors are computed here from real Phase 1/Phase 7 code, not
quoted: the **sensor-noise floor** (Phase 1's own RGB-D noise model applied to
the real GT and scored against it — the error a plausible RealSense/ZED-class
sensor would itself have on this canopy) and the **naive upsampling baseline**
(the same noisy prompt, same 4× downscale, same sky fill, just resized back up
with INTER_NEAREST — the do-nothing alternative PromptDA has to beat).

---

## Findings

1. **DA3 is the strongest pose-conditioned result here, and it wins with less
   conditioning than MapAnything got.** DA3-BASE at 377.2 mm landmark RMSE beats
   MapAnything's 398.8 mm despite having **no depth-conditioning input at all**
   (MapAnything's "full" ingested real metric depth; DA3's cannot). It does so
   with a 0.12B Apache-2.0 checkpoint against MapAnything's 1B CC-BY-NC one, at
   5–7× lower latency per view. DA3-LARGE-1.1 (0.35B, CC-BY-NC) is *not* better
   than DA3-BASE on this axis (398.4 mm), so on this data the licence-clean small
   model is the one to use.

2. **VGGT, with no conditioning at all, matches the pose-conditioned models on
   landmark accuracy** — 381.0 mm images-only, versus DA3's 377.2 mm *with* real
   poses and intrinsics. This was the biggest surprise and it inverts the design
   doc's expectation for the "anchor" model. Two honest qualifications: it is
   measured on 816 landmark-view observations against DA3's 50,734 because
   VGGT's own confidence mask is far stricter, and loosening that mask degrades
   it to 451.6 mm. And its absolute scale is still wrong by 5.2–5.6×, so it is
   accurate in *shape*, not in metric placement. For a pose-known pipeline (which
   this domain is), that scale failure is recoverable and the shape accuracy is
   the valuable part.

3. **`use_ray_pose` did not help on this data, and on the small model it broke
   the evaluation entirely.** DA3's ray head is essentially neutral on DA3-BASE
   (3674.1 → 3679.2 mm camera RMSE) and on DA3-LARGE-1.1 (2507.9 → 2502.5 mm).
   On landmark RMSE it helped DA3-LARGE-1.1 (2849.6 → 2112.9 mm) but on DA3-BASE
   it produced **zero valid landmark observations** — the reconstruction is so
   misaligned that no landmark projects into a mask-valid pixel. The README's
   reported ray-head advantage did not transfer to this thin-canopy domain.

4. **Feeding Pi3X real metric depth made it worse, not better.** Pi3X's
   `pose_intrinsics` arm (poses + intrinsics) scores 1071.8 mm landmark RMSE and
   849,545 occupied voxels; adding real GT depth on top (`full`) *degrades* both
   to 2990.8 mm and 35,912 voxels, and pushes its recovered scale further from
   truth (2.17 → 2.93). Its internode-length error collapses from +0.057 to
   −0.689. This is a real, reproducible, counter-intuitive result on this data
   and is the single most important caveat for anyone planning to feed Pi3X
   sensor depth in this domain.

5. **No unconditioned model recovers metric scale.** Recovered scale factors
   across every images-only run span 2.27×–16.0×. This is uniform across four
   architecturally different models and is the clearest support in this study for
   the design doc's pose-conditioned emphasis.

6. **PromptDA's behaviour is dominated by how no-return regions are encoded, far
   more than by noise or sparsity.** With sky encoded as 0 (PromptDA's own ARKit
   invalid convention — its `load_depth` reads a uint16 PNG /1000 where 0 means
   no return), it collapses completely: predicted depth median ~0.02 m against
   5.4 m truth, AbsRel 0.92, δ1 0.05. Diagnosed directly: 94.6% of the prompt is
   zeros and PromptDA trusts its prompt strongly, so it reproduces them. With the
   sky filled to an assumed 8.0 m background instead (an explicit modelling
   assumption — a real orchard row *has* ground and a next tree row there, Helios
   simply rendered nothing), it recovers **correct metric scale (0.983–0.988)**,
   886–901 mm RMSE and δ1 ≈ 0.89. Against the do-nothing alternative it is a
   genuine win: naive 4× nearest upsampling of the very same prompt scores
   1451–1506 mm, so PromptDA removes ~40% of the error that sparsification
   introduces. It does not get back to the 94–98 mm full-resolution sensor floor.
   *Noise and sparsity themselves barely mattered*: sparse-noisy vs sparse-clean
   vs dense-noisy differ by under 2% RMSE.

7. **MoGe-2's metric scale fails badly on this scene, and known FoV halves but
   does not fix it.** It places the tree 3.7× too far images-only (predicting
   ~19.2 m median where truth is 5.1 m) and 2.1× too far with the real FoV
   supplied (~10.9 m). It also massively over-spreads depth: 15.6 m of predicted
   depth range across a canopy whose real front-to-back extent is 1.22 m
   images-only, improving to 2.9 m with known FoV. δ1 is 0.000 in every arm.
   Even after median scale alignment it is 562–1522 mm RMSE against a 94–98 mm
   sensor floor. A thin tree against featureless sky at 5 m offers almost no
   monocular scale cue, which is a plausible and honest explanation — but the
   practical conclusion stands: **do not use MoGe-2 for metric depth in this
   domain without an external scale reference.**

8. **Everything installed and ran; the only real blocker was one licence wall.**
   Four of five repos needed their pinned torch overridden for sm_120 and all
   worked regardless. No gating on DA3, Pi3X, MoGe-2 or PromptDA checkpoints.
   The single access blocker was `facebook/VGGT-1B-Commercial` (401 GatedRepo),
   which means **the only commercially-licensed checkpoint in this entire study
   is unavailable without HF approval** — DA3-BASE/SMALL (Apache-2.0), MoGe-2
   (MIT) and PromptDA (Apache-2.0) are the licence-clean options that actually
   downloaded.

## What was NOT done

- DA3's **streaming mode** (`da3_streaming/`, sliding-window inference for long
  video) was cloned but not evaluated. These are 16/42-view static rigs, not
  video sequences, so the streaming path had no meaningful input here; it would
  need a different capture (a continuous traversal) to test honestly.
- DA3-GIANT / DA3NESTED-GIANT-LARGE (1.15B/1.40B, CC-BY-NC) not run — the two
  large checkpoints tested (BASE 0.12B, LARGE-1.1 0.35B) already showed the
  larger model was not better on this data, so the marginal value looked low.
- SAM/EfficientSAM — explicitly out of scope per the task (segmentation, not
  reconstruction).
- Base π³ was not run as a fallback because Pi3X installed and ran cleanly, so
  the fallback condition never triggered.
