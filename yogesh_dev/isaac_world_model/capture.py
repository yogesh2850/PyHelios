"""Isaac world-model T2 -- Replicator capture of robot-POV episodes.

Loads orchard_scene.usda (built by build_scene.py) headless, drives a 1.2 m
camera down BOTH 6 m alleys (y = -3 and y = +3) across the ~63 m rows, and
writes per-episode .npz shards in EXACTLY the schema data.py reads:

  rgb       uint8   (T,128,128,3)   area-downsampled from a 256x256 render
  depth     float16 (T,128,128)     metres; sky/background = -1.0 sentinel
  semantic  uint8   (T,128,128)     0 ground/other 1 fruit 2 leaf 3 shoot 6 sky
  instance  int32   (T,128,128)     replicator instance-id ids (0 = none)
  pose      float32 (T,4,4)         camera-to-world (columns right/up/back/pos)
  state     float32 (T,4)           x, y, z, yaw
  a_view    float32 (T,4)           state delta TO THE NEXT frame, last = 0
                                    (the codebase convention: rssm.observe
                                    consumes actions[t] as the action leading
                                    to t+1 -- see actions.states_to_actions)
  a_grow    float32 (T,1)           zeros (static scene)
  fruit_vis float32 (T,)            fraction of fruit-class pixels
  age_days  float32 (T,)            constant 1825

Eight view families: {alley -3, +3} x {direction +x, -x} x {forward-facing,
side-facing at the canopy}, six episodes each with random start / lateral
offset / scan phase -> 48 episodes, split 40/4/4 with every family present in
train and val+test drawn from disjoint families.

--smoke renders 4 hand-picked poses, writes PNG previews (rgb / colorized
semantic / depth) plus smoke_report.json, and exits 0 only if soil, sky, leaf,
shoot and fruit pixels are all present and depth is sane.

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
DATA_OUT = os.path.normpath(os.path.join(HERE, "..", "world_model", "output", "dataset_isaac"))

RENDER_RES = 256
OUT_RES = 128
N_STEPS = 32
CAM_Z = 1.2
PITCH_DEG = -5.0
HFOV_DEG = 60.0
AGE_DAYS = 1825.0
DEPTH_SKY_SENTINEL = -1.0
ROW_HALF_X = 31.5

SEMANTIC_CLASSES = {0: "other/ground", 1: "fruit", 2: "leaf", 3: "shoot",
                    4: "petiole", 5: "peduncle", 6: "sky"}

# family = (name, alley_y, direction, mode, side_yaw)
FAMILIES = [
    ("fwd_m3_px",  -3.0, +1, "fwd",  0.0),
    ("fwd_m3_nx",  -3.0, -1, "fwd",  0.0),
    ("fwd_p3_px",  +3.0, +1, "fwd",  0.0),
    ("fwd_p3_nx",  +3.0, -1, "fwd",  0.0),
    ("side_m3_in", -3.0, +1, "side", +math.pi / 2),
    ("side_m3_out", -3.0, -1, "side", -math.pi / 2),
    ("side_p3_out", +3.0, +1, "side", +math.pi / 2),
    ("side_p3_in", +3.0, -1, "side", -math.pi / 2),
]
EPS_PER_FAMILY = 6
# last episode of each even family -> val, of each odd family -> test
VAL_KEYS = {(0, 5), (2, 5), (4, 5), (6, 5)}
TEST_KEYS = {(1, 5), (3, 5), (5, 5), (7, 5)}

SEM_PALETTE = np.array([[120, 100, 80], [220, 40, 40], [60, 160, 60],
                        [140, 90, 50], [230, 220, 60], [240, 150, 40],
                        [130, 190, 240]], np.uint8)


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def label_to_class(s):
    s = s.lower()
    if "fruit" in s:
        return 1
    if "leaf" in s:
        return 2
    if "shoot" in s or "bark" in s or "trunk" in s:
        return 3
    if "hill" in s or "sky" in s or "background" in s:
        return 6
    return 0  # ground / soil / unlabelled / other


# ---------------------------------------------------------------------------
# trajectories
# ---------------------------------------------------------------------------
def make_trajectory(fam, rng, n_steps=N_STEPS):
    """(T,4) states for one episode of a family."""
    name, y0, direction, mode, side_yaw = fam
    dx = direction * rng.uniform(0.50, 0.70)
    span = abs(dx) * (n_steps - 1)
    if direction > 0:
        x0 = rng.uniform(-30.0, 30.0 - span)
    else:
        x0 = rng.uniform(-30.0 + span, 30.0)
    y_off = rng.uniform(-0.6, 0.6)
    sway_amp = rng.uniform(0.10, 0.30)
    sway_cyc = rng.uniform(1.0, 2.5)
    sway_ph = rng.uniform(0, 2 * math.pi)
    bob_amp = rng.uniform(0.02, 0.06)
    scan_amp = rng.uniform(0.15, 0.28) if mode == "fwd" else rng.uniform(0.08, 0.16)
    scan_cyc = rng.uniform(1.0, 3.0)
    scan_ph = rng.uniform(0, 2 * math.pi)
    yaw0 = (0.0 if direction > 0 else math.pi) if mode == "fwd" else side_yaw

    t = np.arange(n_steps)
    ph = 2 * math.pi * t / n_steps
    x = x0 + dx * t
    y = y0 + y_off + sway_amp * np.sin(sway_cyc * ph + sway_ph)
    z = CAM_Z + bob_amp * np.sin(2.0 * ph + sway_ph)
    yaw = yaw0 + scan_amp * np.sin(scan_cyc * ph + scan_ph) \
        + rng.normal(0, 0.01, n_steps)
    states = np.stack([x, y, z, np.vectorize(wrap_angle)(yaw)], axis=1)
    return states.astype(np.float64)


def states_to_actions(states):
    a = np.zeros_like(states)
    if len(states) > 1:
        a[:-1, :3] = states[1:, :3] - states[:-1, :3]
        a[:-1, 3] = [wrap_angle(d) for d in states[1:, 3] - states[:-1, 3]]
    return a.astype(np.float32)


def state_to_matrix(state):
    """(x,y,z,yaw) -> row-vector-convention Gf-style 4x4 (numpy) camera xform."""
    x, y, z, yaw = [float(v) for v in state]
    p = math.radians(PITCH_DEG)
    d = np.array([math.cos(p) * math.cos(yaw), math.cos(p) * math.sin(yaw),
                  math.sin(p)])
    up_w = np.array([0.0, 0.0, 1.0])
    right = np.cross(d, up_w)
    right /= np.linalg.norm(right)
    up = np.cross(right, d)
    m = np.eye(4)
    m[0, :3] = right          # row-vector convention: rows are camera axes
    m[1, :3] = up
    m[2, :3] = -d             # camera looks down its local -Z
    m[3, :3] = (x, y, z)
    return m


# ---------------------------------------------------------------------------
# capture rig
# ---------------------------------------------------------------------------
class Rig:
    def __init__(self, scene, res, rt_subframes):
        from isaacsim import SimulationApp
        self.app = SimulationApp({"headless": True, "width": res, "height": res})
        import carb.settings
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom
        self.rep, self.Gf, self.UsdGeom = rep, Gf, UsdGeom
        self.rt_subframes = rt_subframes

        # The Fabric Scene Delegate (Isaac 6.0 default) does not resolve
        # SemanticsLabelsAPI on GeomSubsets -- measured: a directly-authored
        # labeled subset renders as UNLABELLED with FSD on. The classic USD
        # scene delegate does resolve them, so turn FSD off before the stage
        # opens.
        carb.settings.get_settings().set_bool("/app/useFabricSceneDelegate", False)

        ok = omni.usd.get_context().open_stage(scene)
        if not ok:
            raise RuntimeError(f"failed to open {scene}")
        self.stage = omni.usd.get_context().get_stage()

        cam = UsdGeom.Camera.Define(self.stage, "/World/RobotCam")
        f = 20.955 / (2.0 * math.tan(math.radians(HFOV_DEG) / 2.0))
        cam.CreateFocalLengthAttr(f)
        cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateVerticalApertureAttr(20.955)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 5000.0))
        xf = UsdGeom.Xformable(cam.GetPrim())
        xf.ClearXformOpOrder()
        self.cam_op = xf.AddTransformOp()
        self.set_pose([0.0, -3.0, CAM_Z, 0.0])

        self.rp = rep.create.render_product("/World/RobotCam", (res, res))
        self.ann = {}
        self.ann["rgb"] = rep.AnnotatorRegistry.get_annotator("rgb")
        self.ann["depth"] = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        self.ann["sem"] = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation", init_params={"colorize": False})
        for name in ("instance_id_segmentation_fast", "instance_id_segmentation"):
            try:
                self.ann["inst"] = rep.AnnotatorRegistry.get_annotator(
                    name, init_params={"colorize": False})
                self.inst_annotator_name = name
                break
            except Exception as e:
                print(f"instance annotator '{name}' unavailable: {e}", flush=True)
        if "inst" not in self.ann:
            raise RuntimeError("no instance-id annotator available")
        for a in self.ann.values():
            a.attach([self.rp])

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
        for _ in range(n):
            self.step(subframes=32)

    @staticmethod
    def _payload(raw):
        """Annotator output -> (array, info-dict)."""
        if isinstance(raw, dict):
            return raw.get("data"), raw.get("info", {}) or {}
        return raw, {}

    def grab(self, state):
        """Render one frame at `state`; return dict of full-res arrays."""
        self.set_pose(state)
        self.step()
        rgb, _ = self._payload(self.ann["rgb"].get_data())
        depth, _ = self._payload(self.ann["depth"].get_data())
        sem_ids, sem_info = self._payload(self.ann["sem"].get_data())
        inst, _ = self._payload(self.ann["inst"].get_data())

        rgb = np.asarray(rgb)[..., :3].astype(np.uint8)
        depth = np.asarray(depth, np.float32).reshape(rgb.shape[0], rgb.shape[1])
        sem_ids = np.asarray(sem_ids).reshape(depth.shape).astype(np.int64)
        inst = np.asarray(inst).reshape(depth.shape).astype(np.int64)

        id2lab = sem_info.get("idToLabels", {})
        lut = np.zeros(max(int(sem_ids.max()), 0) + 1, np.uint8)
        for k, v in id2lab.items():
            try:
                idx = int(k)
            except ValueError:
                continue
            if 0 <= idx < len(lut):
                lut[idx] = label_to_class(json.dumps(v))
        sem = lut[sem_ids]

        nohit = ~np.isfinite(depth) | (depth > 4000.0)
        sem = np.where(nohit, np.uint8(6), sem)
        depth = np.where(sem == 6, np.float32(DEPTH_SKY_SENTINEL), depth)
        return {"rgb": rgb, "depth": depth, "semantic": sem,
                "instance": inst.astype(np.int32)}

    def close(self):
        self.app.close()


def downsample(f):
    """256 -> 128: area-mean rgb, nearest for labels/depth/instance."""
    k = RENDER_RES // OUT_RES
    rgb = f["rgb"].reshape(OUT_RES, k, OUT_RES, k, 3).mean(axis=(1, 3))
    return {"rgb": np.clip(rgb + 0.5, 0, 255).astype(np.uint8),
            "depth": f["depth"][::k, ::k],
            "semantic": f["semantic"][::k, ::k],
            "instance": f["instance"][::k, ::k]}


def frames_to_episode(frames, states):
    a_view = states_to_actions(states)
    sem = np.stack([f["semantic"] for f in frames])
    T = len(frames)
    return {
        "rgb": np.stack([f["rgb"] for f in frames]),
        "depth": np.stack([f["depth"] for f in frames]).astype(np.float16),
        "semantic": sem,
        "instance": np.stack([f["instance"] for f in frames]),
        "pose": np.stack([state_to_matrix(s).T for s in states]).astype(np.float32),
        "state": states.astype(np.float32),
        "a_view": a_view,
        "a_grow": np.zeros((T, 1), np.float32),
        "fruit_vis": (sem == 1).mean(axis=(1, 2)).astype(np.float32),
        "age_days": np.full(T, AGE_DAYS, np.float32),
    }


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------
SMOKE_POSES = [
    ("fwd_alley", (-12.0, -3.0, CAM_Z, 0.0)),
    ("fwd_alley_step", (-11.4, -3.0, CAM_Z, 0.02)),
    ("side_canopy", (2.2, -3.0, CAM_Z, math.pi / 2)),
    ("side_canopy_far", (2.2, 3.0, CAM_Z, -math.pi / 2)),
]


def save_png(path, arr):
    from PIL import Image
    Image.fromarray(arr).save(path)


def depth_vis(d):
    fin = np.isfinite(d) & (d > 0)
    out = np.zeros(d.shape, np.uint8)
    if fin.any():
        v = np.clip(d[fin], 0.1, 60.0)
        out[fin] = (255 - (np.log(v / 0.1) / np.log(600.0)) * 255).astype(np.uint8)
    return out


def run_smoke(rig, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rig.warmup()
    frames = []
    for name, pose in SMOKE_POSES:
        full = rig.grab(np.array(pose))
        f = downsample(full)
        frames.append(f)
        save_png(os.path.join(out_dir, f"smoke_{name}_rgb.png"), full["rgb"])
        save_png(os.path.join(out_dir, f"smoke_{name}_sem.png"),
                 SEM_PALETTE[np.clip(f["semantic"], 0, 6)])
        save_png(os.path.join(out_dir, f"smoke_{name}_depth.png"),
                 depth_vis(f["depth"].astype(np.float32)))

    sem = np.stack([f["semantic"] for f in frames])
    depth = np.stack([f["depth"] for f in frames]).astype(np.float32)
    rgb = np.stack([f["rgb"] for f in frames])
    n = sem.size
    frac = {c: float((sem == c).sum()) / n for c in range(7)}
    fin = depth > 0
    report = {
        "class_fractions": {SEMANTIC_CLASSES[c]: round(frac[c], 5) for c in frac},
        "fruit_pixels": int((sem == 1).sum()),
        "depth_finite_frac": float(fin.mean()),
        "depth_median_m": float(np.median(depth[fin])) if fin.any() else None,
        "depth_max_m": float(depth[fin].max()) if fin.any() else None,
        "rgb_mean": float(rgb.mean()), "rgb_std": float(rgb.std()),
        "instance_annotator": rig.inst_annotator_name,
        "errors": [],
    }
    checks = [
        (frac[0] > 0.02, f"ground/other fraction {frac[0]:.4f} <= 0.02"),
        (frac[6] > 0.02, f"sky fraction {frac[6]:.4f} <= 0.02"),
        (frac[2] > 0.05, f"leaf fraction {frac[2]:.4f} <= 0.05"),
        (frac[3] > 0.001, f"shoot fraction {frac[3]:.4f} <= 0.001"),
        (report["fruit_pixels"] >= 30, f"fruit pixels {report['fruit_pixels']} < 30"),
        (fin.any() and 0.5 < report["depth_median_m"] < 300,
         f"depth median {report['depth_median_m']} outside (0.5, 300)"),
        (20 < report["rgb_mean"] < 235, f"rgb mean {report['rgb_mean']:.1f} outside (20,235)"),
        (report["rgb_std"] > 10, f"rgb std {report['rgb_std']:.1f} <= 10"),
    ]
    for ok, msg in checks:
        if not ok:
            report["errors"].append(msg)
    with open(os.path.join(out_dir, "smoke_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1), flush=True)
    return not report["errors"]


# ---------------------------------------------------------------------------
# full capture
# ---------------------------------------------------------------------------
def run_full(rig, out_root):
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(out_root, split), exist_ok=True)
    rig.warmup()
    episodes = []
    t_start = time.time()
    for fi, fam in enumerate(FAMILIES):
        for ei in range(EPS_PER_FAMILY):
            seed = 1000 + fi * 100 + ei
            rng = np.random.default_rng(seed)
            states = make_trajectory(fam, rng)
            frames = [downsample(rig.grab(s)) for s in states]
            ep = frames_to_episode(frames, states)
            split = ("val" if (fi, ei) in VAL_KEYS
                     else "test" if (fi, ei) in TEST_KEYS else "train")
            rel = os.path.join(split, f"view_f{fi}_{fam[0]}_e{ei}.npz")
            path = os.path.join(out_root, rel)
            np.savez_compressed(path, **ep)
            episodes.append({
                "episode_type": "view", "family": fam[0], "split": split,
                "n_steps": N_STEPS, "resolution": [OUT_RES, OUT_RES],
                "age_days": AGE_DAYS, "seed": seed, "path": rel,
                "alley_y": fam[1], "direction": fam[2], "mode": fam[3],
                "pitch_deg": PITCH_DEG, "hfov_deg": HFOV_DEG,
                "bytes": os.path.getsize(path),
                "fruit_vis_mean": float(ep["fruit_vis"].mean()),
            })
            print(f"[{time.time() - t_start:7.1f}s] {rel}  "
                  f"fruit_vis={ep['fruit_vis'].mean():.4f}", flush=True)

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "source": "isaacsim_replicator",
            "scene": os.path.relpath(SCENE, out_root),
            "stages": [AGE_DAYS],
            "families": [f[0] for f in FAMILIES],
            "n_steps": N_STEPS,
            "resolution": [OUT_RES, OUT_RES],
            "render_resolution": [RENDER_RES, RENDER_RES],
            "semantic_classes": {str(k): v for k, v in SEMANTIC_CLASSES.items()},
            "depth_sky_sentinel": DEPTH_SKY_SENTINEL,
            "hfov_deg": HFOV_DEG, "pitch_deg": PITCH_DEG,
            "camera_height_m": CAM_Z, "age_days": AGE_DAYS,
            "instance_annotator": rig.inst_annotator_name,
        },
        "splits": {s: [e["path"] for e in episodes if e["split"] == s]
                   for s in ("train", "val", "test")},
        "episodes": episodes,
        "totals": {"n_episodes": len(episodes),
                   "n_frames": sum(e["n_steps"] for e in episodes),
                   "bytes": sum(e["bytes"] for e in episodes)},
    }
    with open(os.path.join(out_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote manifest: {manifest['totals']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=SCENE)
    ap.add_argument("--out", default=DATA_OUT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rt-subframes", type=int, default=16)
    args = ap.parse_args()

    rig = Rig(args.scene, RENDER_RES, args.rt_subframes)
    try:
        if args.smoke:
            ok = run_smoke(rig, os.path.join(args.out, "smoke"))
            print("SMOKE_OK" if ok else "SMOKE_FAIL", flush=True)
            code = 0 if ok else 1
        else:
            run_full(rig, args.out)
            print("CAPTURE_DONE", flush=True)
            code = 0
    finally:
        rig.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
