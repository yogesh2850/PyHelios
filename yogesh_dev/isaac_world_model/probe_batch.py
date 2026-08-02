"""Pick the largest training batch size that fits in the CURRENTLY FREE VRAM.

The 256px RSSM shares the 32 GB card with the long-running IsaacLab skrl job,
so "fits in 32 GB" is not the real constraint -- "fits in what skrl left over"
is. Rather than letting the real training OOM (and burn the sampler / class
weight startup each attempt), this builds the exact WorldModel the training run
will build and does two full forward+backward+optimiser steps on synthetic data
at each candidate batch size, largest first. Two steps, not one: the second
step catches the extra allocations Adam's state introduces.

Prints "PROBE_BATCH <n>" on the last line; exits 1 if even batch 1 OOMs.

    /home/yogesh/anaconda3/envs/gsplat/bin/python \
        yogesh_dev/isaac_world_model/probe_batch.py --image-size 256
"""
import argparse
import gc
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from yogesh_dev.world_model.rssm import WorldModel  # noqa: E402


def synthetic_batch(bs, seq, res):
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.5, 60.0, (bs, seq, res, res)).astype(np.float32)
    depth[:, :, :8] = -1.0  # some sky
    return {
        "rgb": rng.integers(0, 256, (bs, seq, res, res, 3), dtype=np.uint8),
        "depth": depth,
        "semantic": rng.integers(0, 7, (bs, seq, res, res), dtype=np.uint8).astype(np.int64),
        "action": rng.normal(0, 0.1, (bs, seq, 5)).astype(np.float32),
        "fruit_vis": rng.uniform(0, 0.05, (bs, seq)).astype(np.float32),
    }


def try_batch(bs, args, device):
    model = WorldModel(action_dim=5, image_size=args.image_size, base=32,
                       deter=512, stoch=32, classes=32, free_bits=12.0,
                       kl_dyn=0.2, kl_rep=0.04, sem_class_weights=[1.0] * 7,
                       depth_loss="l1").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    b = {k: torch.from_numpy(v) for k, v in synthetic_batch(bs, args.seq_len,
                                                            args.image_size).items()}
    for _ in range(2):
        data = model.preprocess(b, device)
        total, _, _, _ = model.loss(data)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        opt.step()
    peak = torch.cuda.max_memory_allocated() / 2**30
    return peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--candidates", default="12,10,8,6,4,3,2,1")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("PROBE_FAIL no cuda")
        sys.exit(1)
    device = "cuda"
    free, total = torch.cuda.mem_get_info()
    print(f"free VRAM at probe time: {free/2**30:.2f} / {total/2**30:.2f} GiB", flush=True)

    for bs in (int(x) for x in args.candidates.split(",")):
        torch.cuda.reset_peak_memory_stats()
        try:
            peak = try_batch(bs, args, device)
            print(f"batch {bs}: OK (peak allocated {peak:.2f} GiB)", flush=True)
            print(f"PROBE_BATCH {bs}")
            return
        except torch.cuda.OutOfMemoryError:
            print(f"batch {bs}: OOM", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    print("PROBE_FAIL even batch 1 OOMs")
    sys.exit(1)


if __name__ == "__main__":
    main()
