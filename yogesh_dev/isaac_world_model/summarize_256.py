"""Compose the isaac_r2 (256px) result: full r1-vs-r2 comparison table +
the one-line Slack message watch_isaac_256.sh sends.

PSNR/SSIM at different resolutions are not directly comparable -- a 256px
image has high-frequency detail a 128px one never had to predict, so r2's
PSNR can be LOWER while the picture is visibly sharper. The table says so and
the sharpness ratio from recon_compare.py is quoted as the direct measure.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WM_OUT = os.path.normpath(os.path.join(HERE, "..", "world_model", "output"))
W6_R2 = os.path.join(WM_OUT, "isaac_r2_w6", "w6_results.json")
W6_R1 = os.path.join(WM_OUT, "isaac_r1_w6", "w6_results.json")
SHARP = os.path.join(WM_OUT, "isaac_r2_w6", "recon_sharpness.json")
TABLE = os.path.join(WM_OUT, "isaac_r2_w6", "compare_r1_r2.txt")
MSG = os.path.join(WM_OUT, "isaac_r2_w6", "slack_msg.txt")

HORIZONS = ("t+1", "t+5", "t+10", "t+25")
METRICS = (("psnr_db", "PSNR dB", "{:6.2f}"), ("ssim", "SSIM", "{:6.4f}"),
           ("depth_mae_m", "depthMAE m", "{:6.3f}"), ("miou", "mIoU", "{:6.4f}"))


def grab(path, variant="full"):
    try:
        r = json.load(open(path))["view"][variant]
        return {h: {m: r[h][m]["mean"] for m, _, _ in METRICS}
                for h in HORIZONS if h in r}
    except Exception as e:
        print(f"warn: {path} ({variant}): {type(e).__name__}: {e}")
        return {}


def main():
    r2, r1 = grab(W6_R2), grab(W6_R1)
    r2_base = grab(W6_R2, "copy_last")
    try:
        sharp = json.load(open(SHARP))
    except Exception:
        sharp = {}

    lines = ["isaac_r2 (256px) vs isaac_r1 (128px), held-out test, open-loop rollout",
             "NOTE: PSNR/SSIM are computed at each model's own resolution; 256px has",
             "high-frequency content 128px never had to predict, so equal-or-lower",
             "PSNR at 256 can still mean a visibly sharper image. Sharpness ratio",
             "(recon |grad| / GT |grad|, native res) is the direct measure.", ""]
    for h in HORIZONS:
        lines.append(f"  {h}")
        for m, label, fmt in METRICS:
            v2 = fmt.format(r2[h][m]) if h in r2 else "   n/a"
            v1 = fmt.format(r1[h][m]) if h in r1 else "   n/a"
            vb = fmt.format(r2_base[h][m]) if h in r2_base else "   n/a"
            lines.append(f"    {label:11s} r2_256={v2}  r1_128={v1}  copy_last_256={vb}")
    if sharp:
        lines.append("")
        lines.append(f"  sharpness (recon-grad/GT-grad): r2_256="
                     f"{sharp.get('r2_grad_ratio_mean', float('nan')):.3f}  "
                     f"r1_128={sharp.get('r1_grad_ratio_mean', float('nan')):.3f}")
    table = "\n".join(lines)
    with open(TABLE, "w") as f:
        f.write(table + "\n")
    print(table)

    def brief(d, h):
        if h not in d:
            return "n/a"
        v = d[h]
        return (f"PSNR {v['psnr_db']:.1f}dB SSIM {v['ssim']:.3f} "
                f"depthMAE {v['depth_mae_m']:.2f}m mIoU {v['miou']:.3f}")

    sh = ""
    if "r2_grad_ratio_mean" in sharp and "r1_grad_ratio_mean" in sharp:
        sh = (f" Sharpness (recon/GT gradient-energy, native res): "
              f"{sharp['r2_grad_ratio_mean']:.2f} vs {sharp['r1_grad_ratio_mean']:.2f} for r1.")
    msg = ("Isaac world-model 256px redo DONE. Same 48-ep/8-family orchard capture "
           "re-rendered at 512 -> area-downsampled to 256 (rt-subframes 48), RSSM "
           "isaac_r2 trained at 256 (isaac_r1 recipe, best-on-val). Test open-loop: "
           f"t+1 {brief(r2, 't+1')}; t+10 {brief(r2, 't+10')}. "
           f"[isaac_r1@128: t+1 {brief(r1, 't+1')}; t+10 {brief(r1, 't+10')} -- "
           "PSNR/SSIM not directly comparable across resolutions]."
           f"{sh} Full table + full-res GT-vs-recon panels + rollout strips in "
           "yogesh_dev/world_model/output/isaac_r2_w6/.")
    with open(MSG, "w") as f:
        f.write(msg)
    print("\n" + msg)


if __name__ == "__main__":
    main()
