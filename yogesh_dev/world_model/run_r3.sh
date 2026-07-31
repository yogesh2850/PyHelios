#!/usr/bin/env bash
# Round 3: attack the reconstruction floor directly, with the escape directions
# the Round 2 sweep already measured. One training run + full evaluation.
#
# The four changes, each justified by a Round 2 measurement:
#
# 1. RESOLUTION 64 -> 128. The npz shards are stored natively at 128x128; Round
#    1/2 downsampled to 64 at load. §12 measured that 97-99% of the t+1 depth
#    error is the autoencoder, §14 that depth is sharpness-limited, and §13/§18
#    that petiole (0.22% of pixels) and peduncle (0.03%) are literally
#    unrepresentable at 64x64, capping mIoU at 5/7 = 0.714. The encoder/decoder
#    scale by log2(image_size)-2 levels, so 128 adds one stride level (5 levels,
#    26.8M params vs 15M). Measured in a smoke run: ~7.9 GB VRAM at batch 16,
#    so batch 24 fits a 32 GB card comfortably.
#
# 2. KL: push past r2_kl. r2_big proved the 1.373-nat KL was a penalty problem,
#    not capacity (4x latent left it unchanged); r2_kl/r2_final at free-bits 6
#    ran the KL at 6.2-6.3 nats -- hugging the floor, i.e. the floor is still
#    the binding constraint. Round 3 raises free-bits 6 -> 12 with the same
#    weights (0.2/0.04). kl_terms clamps the summed-over-categoricals KL per
#    step, so free_bits acts at the total-KL level as intended. The 128px
#    encoder also has 4x the observation to compress, so the latent needs the
#    extra channel more, not less.
#
# 3. Growth action: --growth-subsample becomes the default (r2_growth lifted dt
#    identification from chance 16.9% to 24.5% with it; nothing else moved it at
#    all), and --growth-max-stride is raised 3 -> 6 so a_grow covers
#    5/10/15/20/25/30/35 days (smoke-run histogram: 281/140/88/90/54/60/29).
#    That makes ALL SIX of the growth eval's counterfactual dt values
#    in-distribution, where Round 2 could only train on 5-20 d. The encoding is
#    unchanged (dt_days / 10, recomputed from the stored age_days), which is
#    exactly what run_r2_growth_eval.py feeds at eval time (act[:,:,4] = d/10).
#    --growth-fraction 0.35 compensates for the shorter subsampled windows
#    (r2_growth used 0.40 at stride 3).
#
# 4. Train 40k steps with best-checkpoint-on-val-recon (already implemented in
#    train.py). At 44 orchards the overfitting onset moved to ~26k (§16), and
#    the 128px model is at a lower effective capacity-per-pixel, so 40k with
#    val-every 1000 and a larger --val-batches 16 (the 8-batch val curve
#    oscillates ~7%, §18) gives early-stopping room on both sides.
#
# Kept from r2_final: class-weighted semantic CE (auto), L1 depth.
# NOT run: a 128px no-action ablation (would double GPU time; the zero/shuffled
# action variants inside evaluate.py are the controls this round). The 64px
# r2_noaction checkpoint cannot be used -- its encoder expects 64px input.

set -u
cd /home/yogesh/PyHelios
GSPLAT=/home/yogesh/anaconda3/envs/gsplat/bin/python
WM=yogesh_dev/world_model
DATA=$WM/output/dataset
OUT=$WM/output/r3
LOG=$OUT/r3_pipeline.log
mkdir -p "$OUT"
export PATH=/home/yogesh/anaconda3/envs/gsplat/bin:$PATH
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== R3: 128px + free-bits 12 + growth-subsample stride 6, 40k steps ==="
$GSPLAT -m yogesh_dev.world_model.train --data "$DATA" --steps 40000 \
  --image-size 128 --batch-size 24 --seq-len 32 \
  --growth-fraction 0.35 --growth-subsample --growth-max-stride 6 \
  --free-bits 12.0 --kl-dyn 0.2 --kl-rep 0.04 \
  --sem-class-weights auto --depth-loss l1 --cache-size 1200 \
  --log-every 500 --val-every 1000 --val-batches 16 --ckpt-every 2000 \
  --tag r3_final \
  >> "$WM/output/train_r3_final_stdout.txt" 2>&1
say "r3_final training exit=$?"

say "=== W6 evaluation (held-out test split, 128px) ==="
$GSPLAT -m yogesh_dev.world_model.evaluate \
  --ckpt "$WM/output/train/r3_final/ckpt_best.pt" \
  --data "$DATA" --split test --context 5 --seq-len 32 \
  --batch-size 8 --n-batches 24 --out "$WM/output/r3_w6_final" \
  >> "$OUT/r3_w6_final_stdout.txt" 2>&1
say "evaluation exit=$?"

say "=== reconstruction floor: r2_final (64px) vs r3_final (128px), same seed ==="
$GSPLAT -m yogesh_dev.world_model.run_r2_recon_floor \
  --ckpt "$WM/output/train/r2_final/ckpt_best.pt" --tag r2_final \
  --ckpt "$WM/output/train/r3_final/ckpt_best.pt" --tag r3_final \
  --split test --out "$OUT" --name r3_recon_floor \
  >> "$OUT/r3_recon_floor_stdout.txt" 2>&1
say "recon floor exit=$?"

say "=== growth counterfactual: r2_final, r2_growth, r3_final ==="
# --train-max-dt 35: stride-6 subsampling trains on 5-35 d, so all six
# candidates are in-distribution for r3_final (the restricted sub-metric is
# skipped when it equals the full set; the all-6 accuracy is the common metric).
$GSPLAT -m yogesh_dev.world_model.run_r2_growth_eval --data "$DATA" --split test \
  --ckpt "$WM/output/train/r2_final/ckpt_best.pt"  --tag r2_final \
  --ckpt "$WM/output/train/r2_growth/ckpt_best.pt" --tag r2_growth \
  --ckpt "$WM/output/train/r3_final/ckpt_best.pt"  --tag r3_final \
  --train-max-dt 35 --out "$OUT" \
  >> "$OUT/r3_growth_eval_stdout.txt" 2>&1
say "growth counterfactual exit=$?"

say "=== curves ==="
$GSPLAT -m yogesh_dev.world_model.plot_curves \
  --tags r3_final --out "$OUT" --name r3_curves \
  --title "Round 3: 128px, free-bits 12, growth-subsample stride 6" \
  >> "$OUT/r3_curves_stdout.txt" 2>&1
say "curves exit=$?"
say "ROUND 3 PIPELINE DONE"
