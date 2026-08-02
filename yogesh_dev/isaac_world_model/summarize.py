"""Compose the one-line Slack result message from the W6 eval output."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WM_OUT = os.path.normpath(os.path.join(HERE, "..", "world_model", "output"))
W6 = os.path.join(WM_OUT, "isaac_r1_w6", "w6_results.json")
R3 = os.path.join(WM_OUT, "r3_w6_final", "w6_results.json")
MSG = os.path.join(WM_OUT, "isaac_r1_w6", "slack_msg.txt")


def grab(path, variant="full"):
    try:
        r = json.load(open(path))["view"][variant]
        return {h: (r[h]["psnr_db"]["mean"], r[h]["ssim"]["mean"],
                    r[h]["depth_mae_m"]["mean"], r[h]["miou"]["mean"])
                for h in ("t+1", "t+10") if h in r}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


isaac = grab(W6)
base = grab(W6, "copy_last")
helios = grab(R3)


def fmt(d, h):
    if h not in d:
        return "n/a"
    p, s, dm, iou = d[h]
    return f"PSNR {p:.1f}dB SSIM {s:.3f} depthMAE {dm:.2f}m mIoU {iou:.3f}"


msg = ("Isaac world-model pipeline DONE. 48-ep robot-POV dataset (128px, 8 view "
       "families, soil+sky+hills, 7-class semantics) captured from the 45-tree "
       "Isaac orchard; RSSM isaac_r1 trained (r3 recipe, 30k steps, best-on-val). "
       f"Test open-loop rollout: t+1 {fmt(isaac, 't+1')}; t+10 {fmt(isaac, 't+10')}. "
       f"Copy-last baseline t+10: {fmt(base, 't+10')}. "
       f"[Helios r3_final ref: t+1 {fmt(helios, 't+1')}; t+10 {fmt(helios, 't+10')}]. "
       "Rollout strips + full metrics in yogesh_dev/world_model/output/isaac_r1_w6/.")

with open(MSG, "w") as f:
    f.write(msg)
print(msg)
