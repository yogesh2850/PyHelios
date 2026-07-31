# Foundation-model evaluation: DA3, Pi3X, VGGT, MoGe-2, PromptDA

Real installs, real inference, real numbers on Phase 7's real WAI apple-tree
scenes — scored with the same code paths that produced MapAnything's numbers in
`yogesh_dev/mapanything_eval/`, so the two are directly comparable.

- **`LOG.md`** — what each model is, verified licences, real install commands,
  real blockers, all results, and the findings. Start here.
- **`STATUS.md`** — per-model DONE/BLOCKED verdicts.
- **`output/COMPARISON_TABLES.md`** — machine-generated side-by-side tables
  (including MapAnything's column, read live from its own report JSONs).

## Layout

| file | role |
|---|---|
| `wai_loader.py` | Reads the real WAI scenes (RGB + EXR depth + poses/intrinsics). |
| `prediction_bundle.py` | Shared on-disk schema for the 3D-reconstruction family, byte-compatible with the MapAnything eval's. |
| `depth_bundle.py` | Shared schema + metrics for the depth-focused family. |
| `frame_policy.py` | Decides empirically how each model's output frame relates to the real world frame (the three models behave three different ways — see LOG.md). |
| `run_da3.py`, `run_pi3x.py`, `run_vggt.py` | Inference → reconstruction bundles. |
| `run_moge2.py`, `run_promptda.py` | Inference → depth bundles. |
| `compare_pose_accuracy.py`, `compare_landmark_rmse.py`, `compare_voxel_reconstruction.py` | Reconstruction scoring (ports of the MapAnything eval's scripts, generalized over `--models`). |
| `compare_depth.py` | Per-view depth scoring + two real reference floors. |
| `make_comparison_tables.py` | Emits `output/COMPARISON_TABLES.md` from the report JSONs. |

## Reproducing

Environment setup per model is in `LOG.md` (five dedicated conda envs; every one
overrides its repo's pinned torch with `torch==2.7.0+cu128` for this machine's
RTX 5090 / sm_120). The upstream repos and the ~15 GB of pretrained weights are
gitignored — clone commands are in `LOG.md`.

```bash
export HF_HOME=$PWD/hf_cache

conda activate fm_da3
python run_da3.py --hf_id depth-anything/DA3-BASE      --model_tag base
python run_da3.py --hf_id depth-anything/DA3-LARGE-1.1 --model_tag large11

conda activate fm_pi3   && python run_pi3x.py
conda activate fm_vggt  && python run_vggt.py --hf_id facebook/VGGT-1B --model_tag vggt_1b
conda activate fm_vggt  && python run_vggt.py --hf_id facebook/VGGT-1B --model_tag vggt_1b \
                              --conf_thresh 1.0 --cond_name images_only_laxconf
conda activate fm_depth && python run_moge2.py && python run_promptda.py

conda activate fm_da3
python compare_pose_accuracy.py        --models da3_base da3_large11 pi3x vggt_1b
python compare_landmark_rmse.py        --models da3_base da3_large11 pi3x vggt_1b
python compare_voxel_reconstruction.py --models da3_base da3_large11 pi3x vggt_1b
conda activate fm_depth && python compare_depth.py --models moge2_vitl promptda_vitl
conda activate fm_da3   && python make_comparison_tables.py
```
