# Status — foundation-model evaluation (DA3, Pi3X, VGGT, MoGe-2, PromptDA)

All five requested models were installed and evaluated end to end on the real
Phase 7 WAI apple-tree scenes (`tree0_orbit16` 16 views, `tree_t76_mvs_rig` 42
views), on the RTX 5090, against real Phase 1/Phase 7 ground truth, using the
same scoring code paths that produced MapAnything's published numbers. SAM /
EfficientSAM was excluded as specified (segmentation, out of scope).

Full detail, install commands, blockers and caveats: `LOG.md`.
Machine-generated side-by-side tables: `output/COMPARISON_TABLES.md`.
Raw reports: `output/{pose_accuracy,landmark_rmse,voxel_reconstruction,depth_accuracy}_report.json`.

## Per-model verdicts

### Depth Anything 3 — DONE
Both `DA3-BASE` (Apache-2.0) and `DA3-LARGE-1.1` (CC BY-NC 4.0) run across both
scenes × 3 conditioning levels (`images_only`, `images_only_ray`, `full`). No
blockers. Best pose-conditioned result in the study: **377.2 mm landmark RMSE
(DA3-BASE, full)** vs MapAnything's 398.8 mm, achieved *without* any depth
conditioning (DA3 has no such input) and at 5–7× lower latency per view. The
larger CC-BY-NC checkpoint was not better (398.4 mm), so the licence-clean 0.12B
model is the recommendation. `use_ray_pose` was neutral-to-harmful here.
Caveat recorded: DA3 returns the caller's input extrinsics verbatim under pose
conditioning (verified, residual 4.8e-07), so its pose accuracy is not
measurable through the public API.

### Pi3X — DONE
`yyfz233/Pi3X` run across both scenes × 3 conditioning levels (`images_only`,
`pose_intrinsics`, `full` +depth) — the only model here accepting all three
modalities MapAnything did. No blockers. Weights are CC BY-NC 4.0 (strictly
non-commercial); code is BSD-3. Best result 1071.8 mm landmark RMSE
(`pose_intrinsics`). **Key negative finding: adding real metric depth on top of
poses made it worse** (1071.8 → 2990.8 mm; 849,545 → 35,912 occupied voxels).
Pi3X also keeps its own frame and scale (1.99–2.93× off) even when given real
poses and real metric depth.

### VGGT — DONE, with a real access blocker on the licence-clean checkpoint
Run on both scenes at its only architecturally-supported level (`images_only`,
plus a lax-confidence control arm). **Blocker: `facebook/VGGT-1B-Commercial` is
gated (HTTP 401 GatedRepoError) and could not be downloaded unauthenticated.**
Fell back to the ungated but non-commercial `facebook/VGGT-1B`; results are
valid for research, but the commercially-licensed weights remain unavailable.
Biggest surprise of the study: **381.0 mm landmark RMSE with no conditioning at
all**, matching the pose-conditioned models — though measured on 816
observations vs DA3's 50,734 (its confidence mask keeps only 0.4–5.2% of
pixels), and its absolute scale is still 5.2–5.6× wrong.

### MoGe-2 — DONE
`Ruicheng/moge-2-vitl-normal` (MIT code, ungated weights) run on both scenes at
both levels its API supports (`images_only`, `known_fov`). No blockers.
Evaluated against real per-view EXR depth, as specified. **Negative verdict for
this domain**: metric scale fails badly — the tree is placed 3.7× too far
images-only and 2.1× too far with the real FoV supplied; δ1 = 0.000 in every
arm; even after median scale alignment, 562–1522 mm RMSE against a 94–98 mm real
sensor-noise floor. Not usable for metric depth here without an external scale
reference.

### PromptDA — DONE
`depth-anything/prompt-depth-anything-vitl` (Apache-2.0, ungated) run on both
scenes across four prompt constructions, with the prompt built from real GT
depth via Phase 1's real RGB-D noise model (`apply_rgbd_noise_model`, imported
and called directly, seed 0) applied at full resolution before 4× low-res
readout, as specified. No blockers. **Its behaviour is dominated by how
no-return sky is encoded, not by noise or sparsity**: with sky→0 (PromptDA's own
ARKit invalid convention) it collapses (AbsRel 0.92, δ1 0.05) because 94.6% of
the prompt is zeros; with sky filled to an assumed 8.0 m background it recovers
correct metric scale (0.983–0.988) at 886–901 mm RMSE, δ1 ≈ 0.89, beating naive
4× upsampling of the same prompt (1451–1506 mm) by ~40%. Noise vs no-noise and
sparse vs dense changed RMSE by under 2%.

## Scope not covered (stated explicitly, not silently dropped)

- **DA3 streaming mode** (`da3_streaming/`) cloned but not evaluated — these are
  static 16/42-view rigs, not video sequences, so there was no honest input for
  a sliding-window streaming path. Would need a continuous-traversal capture.
- **DA3-GIANT / DA3NESTED-GIANT-LARGE** not run; the two checkpoints tested
  already showed the larger model was not better on this data.
- **Base π³** not run as a fallback — Pi3X installed and reproduced cleanly, so
  the task's fallback condition never triggered.
- **SAM / EfficientSAM** excluded per the task.

## Constraint compliance

All new files are under `yogesh_dev/foundation_models_eval/`. Nothing outside it
was modified; `helios-core/`, `pyhelios/` and all files with pre-existing
uncommitted changes were untouched. Five dedicated conda envs were created
(`fm_da3`, `fm_pi3`, `fm_vggt`, `fm_depth` shared by MoGe-2 + PromptDA); the
pre-existing `helios`/`gsplat`/`nerf`/`yolo`/`mapanything` envs were not
modified. No `sudo`.

STATUS: DONE
