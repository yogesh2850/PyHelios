# Task: Install MapAnything and run it against the Phase 7 WAI dataset

Unattended headless run (`--dangerously-skip-permissions`), nobody watching interactively.
This task genuinely needs internet access (pip installs, downloading pretrained weights) —
that's expected and allowed here, unlike prior phases.

## Context

Repo: `/home/yogesh/PyHelios`, branch `apple-tree-cameras`. Phase 7 (already done, see
`yogesh_dev/PHASE7_LOG.md`) built a real Helios -> WAI-format dataset writer
(`yogesh_dev/phase7/wai_writer.py`) but found no foundation model (MapAnything/DA3/pi3/VGGT)
installed anywhere on this machine, so it used a classical multi-view-geometry proxy instead
and explicitly flagged every result from that proxy as such. This task closes that gap for
real: install MapAnything and run actual inference against the real WAI dataset already on
disk at `yogesh_dev/phase7/output/wai_dataset/helios_apple_tree/` (two scenes:
`tree0_orbit16`, 16 views; `tree_t76_mvs_rig`, 42 views — real RGB, real EXR depth, real
per-frame pose/intrinsics in each scene's `scene_meta.json`).

**Known real environment facts, verified, don't re-check:**
- `/home/yogesh/anaconda3/envs/gsplat`, `.../nerf`, `.../yolo` all already have working
  torch+CUDA (RTX 5090, compute capability 12.0, recognized and functional) — evidence the
  GPU/driver stack is fine for modern torch. Don't touch these existing envs' packages though
  (see constraint below) — they're this user's working setups for other projects.
- 2.7TB free disk, internet access confirmed (github.com, huggingface.co both reachable).
- One known, real, documented rendering artifact in the source data: foliage pixels in the
  RGB images sometimes saturate to pure white (255,255,255) regardless of flux level (Phase 0
  finding, `yogesh_dev/PHASE0_LOG.md` "photometric saturation artifact") — this affects a
  minority of pixels per frame, not the whole image; don't mistake this for a data-loading bug
  if you see a mostly-white tree region in a rendered frame.

## Constraints

- **Create a new, dedicated conda env for this** (e.g. `conda create -n mapanything python=3.11`,
  or whatever Python version MapAnything's real install instructions specify) rather than
  installing into `helios`/`gsplat`/`nerf`/`yolo` — those are working environments for other
  projects, don't add MapAnything's dependency tree to them and risk version conflicts.
- **Research the actual real install method before assuming a pip package name** — check
  MapAnything's real GitHub repo / official documentation (it's a Meta/FAIR project, released
  2025) for the correct install instructions, model weight source (likely HuggingFace), and
  minimum requirements. Do not guess a package name and let a wrong `pip install` silently
  fail into installing something else.
- Every **new file you create** must live under `/home/yogesh/PyHelios/yogesh_dev/mapanything_eval/`.
  You may install packages into your new conda env (that's expected, not a violation), but do
  not modify anything under `/home/yogesh/PyHelios/` outside that directory, and don't touch
  `helios-core/`, `pyhelios/`, or any file already showing uncommitted changes in `git status`.
  No `git commit`/`push`/`checkout`/`stash`, no `sudo` (a normal user-level conda env doesn't
  need it).
- If MapAnything's real weights are gated behind a HuggingFace login/token that isn't already
  configured on this machine, don't try to work around auth — document that as a real blocker
  and stop cleanly rather than attempting to bypass it.

## What to actually do

1. Find MapAnything's real repo/install instructions and set up a dedicated env for it.
   Confirm the install actually works with a minimal smoke test before moving on.
2. Load the real WAI dataset from `yogesh_dev/phase7/output/wai_dataset/helios_apple_tree/`
   (both scenes) and run real MapAnything inference on it. Test at least two of the four
   conditioning levels Phase 7's D1 ablation used (images-only, and the fullest available
   conditioning e.g. +intrinsics+extrinsics+depth) so there's a real comparison point against
   Phase 7's proxy results for the same conditioning axis.
3. Compare real MapAnything output against:
   - Phase 7's classical-proxy results for the same scene/conditioning
     (`yogesh_dev/phase7/output/t72_pose_conditioning_ablation.json`,
     `t73_baseline_angle_sweep.json`)
   - Real ground truth from Helios (fruit/branch/leaf geometry, camera poses) the same way
     Phase 7's T7.7 did, so you get a real accuracy number, not just a qualitative render.
   - Real wall-clock latency per scene (this was the original motivation — measure it for
     real, don't estimate).
4. Write real output (point clouds / depth predictions / whatever MapAnything actually
   produces) to `yogesh_dev/mapanything_eval/output/`.

## Logging and completion

Continuously append to `yogesh_dev/mapanything_eval/LOG.md`: what MapAnything actually is and
how it was installed, real commands run, real numbers (latency, accuracy vs ground truth,
accuracy vs Phase 7's proxy), any real blockers hit (auth walls, incompatible dependencies,
etc.) and how they were resolved or why they stopped you. When finished (or genuinely
blocked), write `yogesh_dev/mapanything_eval/STATUS.md` whose **last line** is exactly
`STATUS: DONE` or `STATUS: BLOCKED: <reason>` — written last, after everything else.
