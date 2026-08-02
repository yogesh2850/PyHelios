"""Full-resolution single-frame GT-vs-reconstruction comparison, r2 vs r1.

The W6 rollout strips answer "does imagination hold up over horizons"; this
answers the question the 256px redo exists for -- "is the output actually
sharper". For a couple of held-out test episodes it runs the teacher-forced
posterior reconstruction (the model's best case) through BOTH checkpoints and
writes, per chosen frame, a side-by-side panel

    [ GT 256 | isaac_r2 recon 256 | isaac_r1 recon 128 nearest-upscaled x2 ]

plus sharpness numbers: mean absolute image gradient of the reconstruction as
a fraction of the ground truth's, each model measured AT ITS OWN native
resolution (upsampling first would halve r1's gradient energy for free and
rig the comparison).

Runs in the gsplat env. Both datasets use the same trajectory seeds, so
episode files with the same name show the same frames at the two resolutions.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from yogesh_dev.world_model.data import SequenceSampler  # noqa: E402
from yogesh_dev.world_model.evaluate import load_model   # noqa: E402

WM_OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "world_model", "output"))


def grad_energy(img):
    """Mean absolute finite-difference gradient of an RGB float image (H,W,3)."""
    gx = np.abs(np.diff(img, axis=1)).mean()
    gy = np.abs(np.diff(img, axis=0)).mean()
    return float(gx + gy)


@torch.no_grad()
def recon_frames(ckpt, data_root, res, frames, device):
    """Teacher-forced reconstruction -> {path: {t: (gt_u8, recon_u8)}}."""
    model, ck = load_model(ckpt, device)
    assert ck["args"]["image_size"] == res, (ckpt, ck["args"]["image_size"], res)
    sampler = SequenceSampler(data_root, "test", 32, res, growth_fraction=0.0, seed=7)
    out = {}
    for rec, ep in sampler.iter_episodes("view", limit=2, seq_len=32):
        data = model.preprocess({k: torch.from_numpy(v) for k, v in ep.items()}, device)
        state = model.observe(data["obs"], data["action"])
        pred = model.decode(state["h"], state["z"])
        per = {}
        for t in frames:
            gt = ((data["rgb"][0, t] + 0.5).clamp(0, 1) * 255).permute(1, 2, 0)
            pr = ((pred["rgb"][0, t] + 0.5).clamp(0, 1) * 255).permute(1, 2, 0)
            per[t] = (gt.cpu().numpy().astype(np.uint8),
                      pr.cpu().numpy().astype(np.uint8))
        out[os.path.basename(rec["path"])] = per
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r2-ckpt", default=os.path.join(WM_OUT, "train", "isaac_r2", "ckpt_best.pt"))
    ap.add_argument("--r1-ckpt", default=os.path.join(WM_OUT, "train", "isaac_r1", "ckpt_best.pt"))
    ap.add_argument("--data-256", default=os.path.join(WM_OUT, "dataset_isaac_256"))
    ap.add_argument("--data-128", default=os.path.join(WM_OUT, "dataset_isaac"))
    ap.add_argument("--out", default=os.path.join(WM_OUT, "isaac_r2_w6"))
    ap.add_argument("--frames", default="8,24")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = [int(x) for x in args.frames.split(",")]

    r2 = recon_frames(args.r2_ckpt, args.data_256, 256, frames, device)
    r1 = recon_frames(args.r1_ckpt, args.data_128, 128, frames, device)

    from PIL import Image
    stats = {"per_panel": [], "note": "grad_ratio = mean |grad| of recon / mean "
             "|grad| of GT, each model at its native resolution"}
    for name, per in r2.items():
        for t, (gt256, pr256) in per.items():
            panel = [gt256, pr256]
            row = {"episode": name, "frame": int(t),
                   "r2_grad_ratio": grad_energy(pr256 / 255.0) / max(1e-9, grad_energy(gt256 / 255.0))}
            if name in r1 and t in r1[name]:
                gt128, pr128 = r1[name][t]
                row["r1_grad_ratio"] = (grad_energy(pr128 / 255.0)
                                        / max(1e-9, grad_energy(gt128 / 255.0)))
                panel.append(np.repeat(np.repeat(pr128, 2, axis=0), 2, axis=1))
            img = np.concatenate(panel, axis=1)
            fn = f"recon_fullres_{name.replace('.npz', '')}_t{t}.png"
            Image.fromarray(img).save(os.path.join(args.out, fn))
            row["png"] = fn
            stats["per_panel"].append(row)
            print(row, flush=True)

    stats["r2_grad_ratio_mean"] = float(np.mean([r["r2_grad_ratio"] for r in stats["per_panel"]]))
    r1_ratios = [r["r1_grad_ratio"] for r in stats["per_panel"] if "r1_grad_ratio" in r]
    if r1_ratios:
        stats["r1_grad_ratio_mean"] = float(np.mean(r1_ratios))
    with open(os.path.join(args.out, "recon_sharpness.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print("RECON_COMPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
