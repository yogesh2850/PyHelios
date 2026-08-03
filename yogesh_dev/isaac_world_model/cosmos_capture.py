"""Cosmos-Transfer synthetic data capture from the phase9 orchard scene.

Loads orchard_scene.usda headless and drives a robot-POV camera configured to
match a ZED X Mini (2.2 mm wide lens, ~110 deg horizontal FOV, 1.2 m height)
through the two 6 m alleys, writing 121-frame clips with the Replicator
CosmosWriter (RGB + colorized semantic segmentation + shaded seg + depth +
edges, PNG frames plus one mp4 per modality per clip).

Camera reconciliation (documented in COSMOS_DATA.md): the ZED X Mini is
native 1920x1200 (16:10) with ~110 deg H-FOV; Cosmos-Transfer expects
1280x720 (16:9) 121-frame clips. We capture at Cosmos's 720p but with the
ZED's 110 deg horizontal FOV (focal 2.2 mm, horizontal aperture 6.284 mm),
so horizontal geometry matches the real camera; the vertical FOV is ~77.6 deg
instead of the native ~80 deg because of the 16:9 crop.

Trajectories: 14 clips -- 6 passes per alley (both directions, lateral
offsets, gentle S-curves, yaw scans) plus one headland turn-out at each row
end. Segmentation classes: ground, fruit, leaf, shoot, hill (sky = unlabeled
dome light, renders black in the semantic output).

--smoke writes ONE short clip and validates the written PNGs/mp4s
(class fractions, sky presence, image stats); exits non-zero on failure.
--dry-run builds trajectories only (no Isaac) and prints stats.

Run with /home/yogesh/isaacsim/python.sh (Isaac Sim 6.0.1, headless).
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.normpath(os.path.join(HERE, "..", "phase9", "output", "orchard_scene.usda"))
OUT_DEFAULT = os.path.normpath(os.path.join(HERE, "..", "world_model", "output", "cosmos_isaac"))

WIDTH, HEIGHT = 1280, 720            # Cosmos-Transfer input size
N_FRAMES = 121                       # Cosmos-Transfer clip length
FPS = 24.0                           # stage default timeCodesPerSecond -> mp4 fps
CAM_Z = 1.2
PITCH_DEG = -5.0
FOCAL_MM = 2.2                       # ZED X Mini wide lens
HFOV_DEG = 110.0
H_APERTURE = 2.0 * FOCAL_MM * math.tan(math.radians(HFOV_DEG) / 2.0)   # 6.2839 mm
V_APERTURE = H_APERTURE * HEIGHT / WIDTH                               # 3.5347 mm
VFOV_DEG = math.degrees(2.0 * math.atan(V_APERTURE / (2.0 * FOCAL_MM)))
FX_PX = 0.5 * WIDTH / math.tan(math.radians(HFOV_DEG) / 2.0)           # 448.1 px

ROW_HALF_X = 31.5                    # rows span x in [-31.5, 31.5], y in {-6, 0, +6}

# class -> RGBA in the colorized semantic segmentation output. Sky is the
# unlabeled dome light and comes out (0, 0, 0); Cosmos-Transfer's seg control
# only needs the colors to be consistent, not any particular palette.
SEG_MAPPING = {
    "ground": [110, 70, 35, 255],
    "fruit":  [230, 30, 30, 255],
    "leaf":   [40, 170, 40, 255],
    "shoot":  [200, 140, 90, 255],
    "hill":   [120, 120, 160, 255],
}
SKY_RGB = (0, 0, 0)


def state_to_matrix(state):
    """(x,y,z,yaw) -> row-vector-convention 4x4 camera xform (as capture.py)."""
    x, y, z, yaw = [float(v) for v in state]
    p = math.radians(PITCH_DEG)
    d = np.array([math.cos(p) * math.cos(yaw), math.cos(p) * math.sin(yaw),
                  math.sin(p)])
    up_w = np.array([0.0, 0.0, 1.0])
    right = np.cross(d, up_w)
    right /= np.linalg.norm(right)
    up = np.cross(right, d)
    m = np.eye(4)
    m[0, :3] = right
    m[1, :3] = up
    m[2, :3] = -d
    m[3, :3] = (x, y, z)
    return m


# ---------------------------------------------------------------------------
# trajectories
# ---------------------------------------------------------------------------
def resample_path(points, n):
    """Dense polyline -> n points at constant arc-length spacing."""
    pts = np.asarray(points, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    si = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(si, s, pts[:, 0]), np.interp(si, s, pts[:, 1])], 1)


def line_path(x0, x1, y, lat_off=0.0, sway_amp=0.0, sway_cyc=1.0, sway_ph=0.0, m=600):
    t = np.linspace(0.0, 1.0, m)
    x = x0 + (x1 - x0) * t
    yy = y + lat_off + sway_amp * np.sin(2 * math.pi * sway_cyc * t + sway_ph)
    return np.stack([x, yy], 1)


def headland_path(end_sign, y_from, y_to, r=3.0, lead=2.0, m=900):
    """Straight -> half-circle around the row end -> straight into other alley."""
    cx = end_sign * (ROW_HALF_X + 1.5)
    t = np.linspace(0.0, 1.0, m // 3)
    p1 = np.stack([np.full_like(t, 0), np.full_like(t, y_from)], 1)
    p1[:, 0] = (cx - end_sign * lead) + end_sign * lead * t
    sweep = np.linspace(-math.pi / 2, -math.pi / 2 + end_sign * math.pi, m // 3)
    arc = np.stack([cx + r * np.cos(sweep) * end_sign * np.sign(y_to - y_from),
                    r * np.sin(sweep)], 1)
    # orient the arc so it starts at y_from and bulges past the row end
    if abs(arc[0, 1] - y_from) > 1e-6:
        arc = arc[::-1]
    p3 = np.stack([cx - end_sign * lead * t, np.full_like(t, y_to)], 1)
    return np.concatenate([p1, arc, p3], 0)


def build_states(path_xy, n_frames, scan_amp=0.0, scan_cyc=1.0, scan_ph=0.0,
                 bob_amp=0.03, seed=0):
    """Path -> (T,4) states with tangent yaw + scan sinusoid + height bob."""
    xy = resample_path(path_xy, n_frames)
    d = np.gradient(xy, axis=0)
    yaw = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    rng = np.random.default_rng(seed)
    ph = 2 * math.pi * np.arange(n_frames) / n_frames
    yaw = yaw + scan_amp * np.sin(scan_cyc * ph + scan_ph) + rng.normal(0, 0.004, n_frames)
    z = CAM_Z + bob_amp * np.sin(2.0 * ph + scan_ph)
    return np.stack([xy[:, 0], xy[:, 1], z, yaw], 1)


# name, path, scan_amp, scan_cyc, scan_ph  (alley A: y=-3, alley B: y=+3)
CLIP_SPECS = [
    ("A_fwd_center", line_path(-28, -20, -3, 0.0, 0.15, 1.5, 0.3), 0.06, 1.0, 0.2),
    ("A_fwd_latP",   line_path(-6, 3, -3, +0.8, 0.10, 1.0, 1.1),   0.12, 1.5, 1.0),
    ("A_rev_latN",   line_path(24, 16, -3, -0.7, 0.12, 2.0, 2.0),  0.08, 1.0, 2.1),
    ("A_rev_scan",   line_path(8, 1, -3, 0.0, 0.08, 1.0, 0.5),     0.35, 2.0, 0.7),
    ("A_fwd_curve",  line_path(12, 21, -3, 0.0, 0.80, 1.5, 0.0),   0.05, 1.0, 1.4),
    ("A_rev_center", line_path(-14, -23, -3, 0.0, 0.25, 2.5, 1.7), 0.07, 1.0, 2.8),
    ("B_fwd_latN",   line_path(-30, -21, 3, -0.8, 0.12, 1.5, 0.9), 0.06, 1.0, 0.4),
    ("B_fwd_scan",   line_path(2, 11, 3, 0.0, 0.10, 1.0, 2.4),     0.30, 1.5, 1.9),
    ("B_rev_center", line_path(28, 20, 3, 0.0, 0.15, 2.0, 0.2),    0.05, 1.0, 2.5),
    ("B_rev_latP",   line_path(-2, -11, 3, +0.7, 0.20, 1.5, 1.3),  0.08, 1.0, 0.9),
    ("B_fwd_curve",  line_path(-18, -9, 3, 0.0, 0.80, 2.0, 0.6),   0.04, 1.0, 1.6),
    ("B_rev_mild",   line_path(18, 9, 3, 0.0, 0.10, 1.0, 2.9),     0.10, 2.0, 2.2),
    ("H_east_AtoB",  headland_path(+1, -3, +3),                    0.00, 1.0, 0.0),
    ("H_west_BtoA",  headland_path(-1, +3, -3),                    0.00, 1.0, 0.0),
]

SMOKE_SPEC = ("smoke_A_fwd", line_path(-12.0, -11.4, -3, 0.0, 0.05, 1.0, 0.0),
              0.05, 1.0, 0.0)
SMOKE_FRAMES = 12


def clip_states(spec, n_frames, seed):
    name, path, amp, cyc, ph = spec
    return name, build_states(path, n_frames, amp, cyc, ph, seed=seed)


# ---------------------------------------------------------------------------
# capture rig
# ---------------------------------------------------------------------------
class CosmosRig:
    def __init__(self, scene, out_dir, rt_subframes):
        from isaacsim import SimulationApp
        self.app = SimulationApp({"headless": True, "width": WIDTH, "height": HEIGHT})
        import carb.settings
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom
        self.rep, self.Gf = rep, Gf
        self.rt_subframes = rt_subframes

        settings = carb.settings.get_settings()
        # FSD does not resolve SemanticsLabelsAPI on GeomSubsets (measured in
        # the earlier capture work); the classic delegate does.
        settings.set_bool("/app/useFabricSceneDelegate", False)
        # CosmosWriter requires script nodes
        settings.set_bool("/app/omni.graph.scriptnode/opt_in", True)
        settings.set("rtx/post/dlss/execMode", 2)
        rep.orchestrator.set_capture_on_play(False)

        ok = omni.usd.get_context().open_stage(scene)
        if not ok:
            raise RuntimeError(f"failed to open {scene}")
        self.stage = omni.usd.get_context().get_stage()

        cam = UsdGeom.Camera.Define(self.stage, "/World/CosmosCam")
        cam.CreateFocalLengthAttr(FOCAL_MM)
        cam.CreateHorizontalApertureAttr(H_APERTURE)
        cam.CreateVerticalApertureAttr(V_APERTURE)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 5000.0))
        xf = UsdGeom.Xformable(cam.GetPrim())
        xf.ClearXformOpOrder()
        self.cam_op = xf.AddTransformOp()
        self.set_pose([0.0, -3.0, CAM_Z, 0.0])
        print(f"camera: focal {FOCAL_MM} mm, hAperture {H_APERTURE:.4f} mm "
              f"-> HFOV {HFOV_DEG:.1f} deg, VFOV {VFOV_DEG:.1f} deg, "
              f"fx {FX_PX:.1f} px @ {WIDTH}x{HEIGHT}", flush=True)

        self.rp = rep.create.render_product("/World/CosmosCam", (WIDTH, HEIGHT))
        self.writer = None
        self.out_dir = out_dir

    def attach_writer(self):
        rep = self.rep
        backend = rep.backends.get("DiskBackend")
        backend.initialize(output_dir=self.out_dir)
        self.writer = rep.WriterRegistry.get("CosmosWriter")
        self.writer.initialize(backend=backend, segmentation_mapping=SEG_MAPPING)
        self.writer.attach(self.rp)

    def set_pose(self, state):
        m = state_to_matrix(state)
        self.cam_op.Set(self.Gf.Matrix4d(*m.flatten().tolist()))

    def step(self, subframes=None):
        sf = self.rt_subframes if subframes is None else subframes
        try:
            self.rep.orchestrator.step(rt_subframes=sf, delta_time=0.0,
                                       pause_timeline=True)
        except TypeError:
            self.rep.orchestrator.step(rt_subframes=sf)

    def warmup(self, n=12):
        # before the writer attaches, so warmup frames are not written
        for _ in range(n):
            self.step(subframes=32)

    def capture_clip(self, name, states, clip_index, n_total):
        t0 = time.time()
        for i, s in enumerate(states):
            self.set_pose(s)
            self.step()
            if (i + 1) % 20 == 0:
                print(f"  clip {clip_index + 1}/{n_total} [{name}] "
                      f"frame {i + 1}/{len(states)} "
                      f"({(time.time() - t0) / (i + 1):.2f} s/frame)", flush=True)
        self.writer.next_clip()   # encodes this clip's mp4s, advances index
        print(f"clip {clip_index + 1}/{n_total} [{name}] done in "
              f"{time.time() - t0:.1f} s", flush=True)

    def finish(self):
        self.rep.orchestrator.wait_until_complete()
        if self.writer is not None:
            self.writer.detach()
        self.rp.destroy()

    def close(self):
        self.app.close()


# ---------------------------------------------------------------------------
# validation of written output (plain PIL/numpy, no Isaac needed)
# ---------------------------------------------------------------------------
def validate_clip(clip_dir, n_frames, report_path):
    from PIL import Image

    report = {"clip_dir": clip_dir, "errors": [], "warnings": []}
    counts = {}
    for mod in ("rgb", "segmentation", "shaded_seg", "depth", "edges"):
        d = os.path.join(clip_dir, mod)
        counts[mod] = len([f for f in os.listdir(d) if f.endswith(".png")]) if os.path.isdir(d) else 0
        if counts[mod] != n_frames:
            report["errors"].append(f"{mod}: {counts[mod]} pngs, expected {n_frames}")
    report["png_counts"] = counts
    mp4s = [f for f in os.listdir(clip_dir) if f.endswith(".mp4")] if os.path.isdir(clip_dir) else []
    report["mp4s"] = sorted(mp4s)
    if len(mp4s) < 5:
        report["errors"].append(
            f"only {len(mp4s)} mp4s in {clip_dir}, expected 5 (NVENC hardware "
            f"encoding failed or unavailable?)")

    if counts.get("rgb") and counts.get("segmentation"):
        rgbs, segs = [], []
        for i in range(min(n_frames, counts["rgb"])):
            rgbs.append(np.asarray(Image.open(
                os.path.join(clip_dir, "rgb", f"rgb_{i:04}.png")))[..., :3])
            segs.append(np.asarray(Image.open(
                os.path.join(clip_dir, "segmentation", f"segmentation_{i:04}.png")))[..., :3])
        rgb = np.stack(rgbs)
        seg = np.stack(segs)
        if rgb.shape[1:3] != (HEIGHT, WIDTH):
            report["errors"].append(f"rgb resolution {rgb.shape[1:3]} != {(HEIGHT, WIDTH)}")

        n = float(seg[..., 0].size)
        frac = {}
        matched = np.zeros(seg.shape[:3], bool)
        for cls, rgba in SEG_MAPPING.items():
            m = np.all(seg == np.array(rgba[:3], np.uint8), axis=-1)
            frac[cls] = float(m.sum()) / n
            matched |= m
        sky_frac = float((~matched).sum()) / n
        frac["sky/unlabeled"] = sky_frac
        top = ~matched[:, : HEIGHT // 5, :]
        report["class_fractions"] = {k: round(v, 5) for k, v in frac.items()}
        report["fruit_pixels"] = int(frac["fruit"] * n)
        report["top_band_sky_frac"] = round(float(top.mean()), 4)
        report["rgb_mean"] = round(float(rgb.mean()), 2)
        report["rgb_std"] = round(float(rgb.std()), 2)

        checks = [
            (frac["ground"] > 0.03, f"ground fraction {frac['ground']:.4f} <= 0.03"),
            (frac["leaf"] > 0.03, f"leaf fraction {frac['leaf']:.4f} <= 0.03"),
            (frac["shoot"] > 0.0005, f"shoot fraction {frac['shoot']:.5f} <= 0.0005"),
            (report["fruit_pixels"] > 100, f"fruit pixels {report['fruit_pixels']} <= 100"),
            (frac["hill"] > 0.0002, f"hill fraction {frac['hill']:.5f} <= 0.0002 "
             "(distant hills not visible?)"),
            (sky_frac > 0.01, f"sky/unlabeled fraction {sky_frac:.4f} <= 0.01"),
            (15 < report["rgb_mean"] < 240, f"rgb mean {report['rgb_mean']} outside (15,240)"),
            (report["rgb_std"] > 10, f"rgb std {report['rgb_std']} <= 10"),
        ]
        for ok, msg in checks:
            if not ok:
                report["errors"].append(msg)
        if report["top_band_sky_frac"] < 0.05:
            report["warnings"].append(
                f"top-band sky fraction {report['top_band_sky_frac']} < 0.05 "
                "(expected sky near top of frame)")

    with open(report_path, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1), flush=True)
    return not report["errors"]


def camera_meta():
    return {
        "model": "ZED X Mini (simulated)",
        "resolution": [WIDTH, HEIGHT],
        "focal_length_mm": FOCAL_MM,
        "horizontal_aperture_mm": round(H_APERTURE, 4),
        "vertical_aperture_mm": round(V_APERTURE, 4),
        "hfov_deg": HFOV_DEG,
        "vfov_deg": round(VFOV_DEG, 2),
        "fx_px": round(FX_PX, 2), "fy_px": round(FX_PX, 2),
        "cx_px": WIDTH / 2, "cy_px": HEIGHT / 2,
        "height_m": CAM_Z, "pitch_deg": PITCH_DEG, "fps": FPS,
        "note": "ZED native 1920x1200/16:10; captured at Cosmos 1280x720/16:9 "
                "with ZED 110deg H-FOV, so VFOV is 77.6 vs native ~80 deg",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=SCENE)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="build trajectories only (no Isaac), print stats")
    ap.add_argument("--rt-subframes", type=int, default=16)
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    args = ap.parse_args()

    if args.dry_run:
        for i, spec in enumerate(CLIP_SPECS):
            name, st = clip_states(spec, args.frames, seed=4000 + i)
            step = np.linalg.norm(np.diff(st[:, :2], axis=0), axis=1)
            print(f"{name:14s} frames={len(st)} span={step.sum():6.2f} m "
                  f"speed={step.mean() * FPS:4.2f} m/s "
                  f"x[{st[:, 0].min():6.1f},{st[:, 0].max():6.1f}] "
                  f"y[{st[:, 1].min():5.2f},{st[:, 1].max():5.2f}] "
                  f"yaw[{math.degrees(st[:, 3].min()):7.1f},"
                  f"{math.degrees(st[:, 3].max()):7.1f}]deg")
            assert np.all(np.abs(st[:, 1]) < 5.0) or name.startswith("H_"), name
        print("DRY_RUN_OK")
        return

    os.makedirs(args.out, exist_ok=True)
    rig = CosmosRig(args.scene, args.out, args.rt_subframes)
    code = 1
    try:
        rig.warmup()
        rig.attach_writer()
        if args.smoke:
            name, states = clip_states(SMOKE_SPEC, SMOKE_FRAMES, seed=7)
            rig.capture_clip(name, states, 0, 1)
            rig.finish()
            ok = validate_clip(os.path.join(args.out, "clip_0000"), SMOKE_FRAMES,
                               os.path.join(args.out, "smoke_report.json"))
            print("SMOKE_OK" if ok else "SMOKE_FAIL", flush=True)
            code = 0 if ok else 1
        else:
            manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "scene": os.path.relpath(args.scene, args.out),
                        "camera": camera_meta(),
                        "n_frames_per_clip": args.frames,
                        "segmentation_mapping": SEG_MAPPING,
                        "sky_rgb": list(SKY_RGB),
                        "clips": []}
            for i, spec in enumerate(CLIP_SPECS):
                name, states = clip_states(spec, args.frames, seed=4000 + i)
                rig.capture_clip(name, states, i, len(CLIP_SPECS))
                step = np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1)
                manifest["clips"].append({
                    "dir": f"clip_{i:04}", "name": name,
                    "frames": len(states),
                    "path_span_m": round(float(step.sum()), 2),
                    "mean_speed_mps": round(float(step.mean() * FPS), 2),
                })
            rig.finish()
            with open(os.path.join(args.out, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=1)
            # validate the first and last clip (counts + content)
            ok = all(validate_clip(os.path.join(args.out, f"clip_{i:04}"), args.frames,
                                   os.path.join(args.out, f"validate_clip_{i:04}.json"))
                     for i in (0, len(CLIP_SPECS) - 1))
            print("CAPTURE_OK" if ok else "CAPTURE_VALIDATION_FAIL", flush=True)
            code = 0 if ok else 1
    finally:
        rig.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
