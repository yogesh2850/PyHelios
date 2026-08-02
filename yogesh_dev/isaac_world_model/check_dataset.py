"""Post-capture dataset sanity gate (runs in the gsplat env).

Verifies the Isaac dataset is loadable by the EXISTING data.py before burning
GPU-hours on training: schema/dtype/shape of every field, manifest counts,
action convention (a_view replays states), class coverage, and one real
SequenceSampler batch.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "world_model", "output", "dataset_isaac")
ROOT = os.path.normpath(ROOT)

errors = []


def check(cond, msg):
    if not cond:
        errors.append(msg)
        print("FAIL:", msg)


m = json.load(open(os.path.join(ROOT, "manifest.json")))
eps = m["episodes"]
check(len(eps) == 48, f"episode count {len(eps)} != 48")
by_split = {s: sum(1 for e in eps if e["split"] == s) for s in ("train", "val", "test")}
check(by_split == {"train": 40, "val": 4, "test": 4}, f"splits {by_split}")

want = {"rgb": (np.uint8, (32, 128, 128, 3)), "depth": (np.float16, (32, 128, 128)),
        "semantic": (np.uint8, (32, 128, 128)), "instance": (np.int32, (32, 128, 128)),
        "pose": (np.float32, (32, 4, 4)), "state": (np.float32, (32, 4)),
        "a_view": (np.float32, (32, 4)), "a_grow": (np.float32, (32, 1)),
        "fruit_vis": (np.float32, (32,)), "age_days": (np.float32, (32,))}
cls_pix = np.zeros(7)
for e in eps[::7] + [eps[-1]]:
    with np.load(os.path.join(ROOT, e["path"])) as z:
        for k, (dt, shp) in want.items():
            check(k in z.files, f"{e['path']}: missing {k}")
            check(z[k].dtype == dt, f"{e['path']}:{k} dtype {z[k].dtype} != {dt}")
            check(z[k].shape == shp, f"{e['path']}:{k} shape {z[k].shape} != {shp}")
        # action convention: replaying a_view from state[0] must land on state[t]
        s, a = z["state"].astype(np.float64), z["a_view"].astype(np.float64)
        rs = s[0].copy()
        for t in range(1, len(s)):
            rs[:3] += a[t - 1, :3]
            rs[3] += a[t - 1, 3]
            err = np.abs(rs[:3] - s[t, :3]).max()
            check(err < 1e-4, f"{e['path']}: action replay error {err:.2e} at t={t}")
            if err >= 1e-4:
                break
        cls_pix += np.bincount(z["semantic"].reshape(-1), minlength=7)
        d = z["depth"].astype(np.float32)
        sky = z["semantic"] == 6
        check((d[sky] == -1.0).all() if sky.any() else True,
              f"{e['path']}: sky depth not sentinel")
        check(np.isfinite(d[~sky]).all() and (d[~sky] > 0).all(),
              f"{e['path']}: non-sky depth not positive/finite")

frac = cls_pix / cls_pix.sum()
print("class fractions over sampled episodes:", np.round(frac, 5).tolist())
for c, name, lo in ((0, "ground/other", 0.05), (1, "fruit", 0.0005),
                    (2, "leaf", 0.02), (3, "shoot", 0.002), (6, "sky", 0.02)):
    check(frac[c] > lo, f"class {name} fraction {frac[c]:.5f} <= {lo}")

sys.path.insert(0, os.path.normpath(os.path.join(ROOT, "..", "..", "..", "..")))
from yogesh_dev.world_model.data import SequenceSampler  # noqa: E402

smp = SequenceSampler(ROOT, "train", seq_len=32, image_size=128, growth_fraction=0.0)
b = smp.sample_batch(4)
print("sample_batch shapes:", {k: tuple(v.shape) for k, v in b.items()})
check(b["action"].shape == (4, 32, 5), f"action shape {b['action'].shape}")
check(b["rgb"].shape == (4, 32, 128, 128, 3), f"rgb batch shape {b['rgb'].shape}")

if errors:
    print(f"DATASET_CHECK_FAIL ({len(errors)} errors)")
    sys.exit(1)
print("DATASET_CHECK_OK")
