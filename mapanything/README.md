# MapAnything evaluation

Real installation and inference of [MapAnything](https://github.com/facebookresearch/map-anything)
(`facebook/map-anything` weights) against the Phase 7 WAI-format dataset
(`yogesh_dev/phase7/output/wai_dataset/`), closing the gap Phase 7 flagged: no foundation
model was installed anywhere on this machine at the time.

See `STATUS.md` for the outcome and `LOG.md` for full detail (install steps, real commands,
real numbers). Short version: no blockers, ~0.12-0.13s/view real inference latency on the
RTX 5090, full pose conditioning far outperforms images-only, and results are mixed vs. the
Phase 7 classical baseline — better on thin-structure/internode metrics, worse on landmark
accuracy and fruit-diameter accuracy. `viz/comparison_3methods.png` shows MapAnything vs. the
classical baseline vs. real ground truth side by side on the same tree.

## Layout
- `wai_loader.py`, `run_inference.py`, `export_pointclouds.py`, `compare_*.py` — the actual
  evaluation code.
- `output/` — real results: point clouds (`.ply`/`.npy`), raw predictions (`.npz`), and
  comparison reports (`.json`) for both scenes x both conditioning levels.
- `viz/` — point-cloud renders, including a 3-way comparison against the classical baseline
  and ground truth.

## Not included here (reproducible, not duplicated)
- `map-anything/` — the cloned upstream repo (~20MB). Re-clone from
  `https://github.com/facebookresearch/map-anything`.
- `hf_cache/` — downloaded model weights (~4.6GB) from the `facebook/map-anything` HuggingFace
  repo. Re-downloads automatically on first inference run.
- The dedicated `mapanything` conda env (python 3.12, torch 2.7.0+cu128) — see `LOG.md` for
  the exact install steps used to recreate it.
