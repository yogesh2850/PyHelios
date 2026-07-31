# Task: Install and evaluate 4 more foundation models (DA3, Pi3X, VGGT, MoGe-2, PromptDA)

Unattended headless run (`--dangerously-skip-permissions`), nobody watching interactively.
This task genuinely needs internet access (pip installs, downloading pretrained weights) —
expected and allowed.

## Context

Repo: `/home/yogesh/PyHelios`, branch `apple-tree-cameras`. This follows directly from the
MapAnything evaluation already done (`yogesh_dev/mapanything_eval/` worktree branch
`worktree-mapanything-eval-real`, also copied to `/home/yogesh/PyHelios/mapanything/` — read
`LOG.md` there for the harness pattern to reuse: it loads the real WAI dataset, runs
inference, and compares against ground truth and a classical baseline).

A research design document (attached context, not a file — key excerpt below) recommends a
specific set of foundation models to evaluate for this pose-known, thin-structure, apple-canopy
domain. The user wants **all of them tried except SAM** (SAM/EfficientSAM is segmentation, not
reconstruction — explicitly out of scope for this task). That's 5 models across two families:

**3D reconstruction family (compare like MapAnything was compared):**
- **Depth Anything 3 (DA3)** — has an explicit pose-conditioned mode (`use_ray_pose`) and a
  streaming mode. Real repo, Apache-2.0 for BASE/SMALL/METRIC-LARGE/MONO-LARGE weight variants
  (verify the exact license per checkpoint before downloading a NC one).
- **Pi3X** (or base π³ if Pi3X isn't cleanly findable/reproducible — document which) — optional
  pose/intrinsics/depth conditioning. Code is BSD-3, **weights are CC-BY-NC** — fine for this
  research evaluation, just don't present it as commercially usable without flagging that.
- **VGGT** — no pose-conditioning input (uncalibrated only). Include as a comparison anchor per
  the design doc, budget less time on it than the pose-conditioned models. `VGGT-1B-Commercial`
  checkpoint is the license-clean one if available.

**Depth-focused family (different evaluation — compare against real per-view ground-truth
depth, not full reconstruction):**
- **MoGe-2** — metric monocular depth, MIT license, fast.
- **PromptDA** — RGB + sparse/noisy depth -> dense metric depth completion, Apache-2.0. Feed it
  sparse/noisy depth by subsampling and adding noise to Phase 1's real ground-truth EXR depth
  (Phase 1's `yogesh_dev/phase1/noise_model.py` already has a real RGB-D noise model — reuse it
  rather than inventing a new noise process) to simulate what a real depth sensor would give it,
  then check how well it recovers the real dense ground truth.

**Verify each model's real repo, install instructions, and exact license before downloading
anything** — do not guess package names or assume a license from memory. If a model turns out
gated behind an unavailable auth wall or genuinely can't be installed, document that as a real,
honest per-model blocker and move on to the next one rather than stalling the whole task.

Real data already available, reuse it:
- `yogesh_dev/phase7/output/wai_dataset/helios_apple_tree/` — two scenes (`tree0_orbit16` 16
  views, `tree_t76_mvs_rig` 42 views), real RGB, real EXR depth, real poses/intrinsics.
- `yogesh_dev/phase1/output/` and `yogesh_dev/phase7/output/` — real ground truth (fruit
  positions, branch segments, canopy geometry) to evaluate against, same as MapAnything's eval
  used.

## Constraints

- **One dedicated conda env per model family** (or per model if dependencies conflict) — do
  not install into `helios`/`gsplat`/`nerf`/`yolo`/`mapanything`, those are working
  environments for other things. 2.7TB free disk, that's not a constraint here.
- Every **new file you create** must live under `/home/yogesh/PyHelios/yogesh_dev/foundation_models_eval/`.
  Do not modify anything under `/home/yogesh/PyHelios/` outside that directory, don't touch
  `helios-core/`, `pyhelios/`, or any file already showing uncommitted changes in git status.
  No `git commit`/`push`/`checkout`/`stash`, no `sudo`.
- Budget your effort — 5 models is a lot. If time/budget runs out, finish whichever models are
  done cleanly and clearly report which ones weren't reached, rather than doing all 5 shallowly
  and none well. Prioritize DA3 first (most directly comparable to MapAnything, has the
  streaming angle the design doc cares about), then Pi3X, then VGGT, then MoGe-2/PromptDA.

## What to actually do, per model

For the 3D-reconstruction family (DA3, Pi3X, VGGT): reuse the same evaluation methodology as
`yogesh_dev/mapanything_eval/compare_pose_accuracy.py` / `compare_landmark_rmse.py` /
`compare_voxel_reconstruction.py` (read them for the pattern) so results are directly
comparable to MapAnything's real numbers (landmark RMSE 4.6mm classical / 398.8mm MapAnything
full-conditioning; thin-structure recall by diameter class; real latency). Run both scenes,
test at least images-only and full-conditioning where the model supports it (skip conditioning
levels a model architecturally doesn't support and say so).

For the depth-focused family (MoGe-2, PromptDA): compare predicted depth against Phase 1's
real EXR ground truth depth, per-pixel and per-view, real RMSE, plus real latency.

## Logging and completion

Continuously append to `yogesh_dev/foundation_models_eval/LOG.md`: what each model actually
is, real install steps, real commands run, real numbers per model, and — critically — a
running comparison table against MapAnything's already-known real numbers so the reader can
see all models side by side. Any real per-model blocker (license wall, broken install, etc.)
goes in the log with what was tried. When finished (or the budget runs out), write
`yogesh_dev/foundation_models_eval/STATUS.md` whose **last line** is exactly `STATUS: DONE` or
`STATUS: BLOCKED: <reason>` — written last. Report DONE/BLOCKED **per model**, not just one
overall verdict, since partial completion across 5 models is a realistic and acceptable outcome.
