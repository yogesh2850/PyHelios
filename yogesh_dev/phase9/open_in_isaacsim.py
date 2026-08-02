"""Phase 9 / T9.3 — open a PyHelios-exported USD in a real Isaac Sim session and render it.

Runs in the `env_isaaclab` conda env (Isaac Sim 5.1.0 pip package), where `pxr` and the Kit
runtime live. `pxr` is only importable *after* `SimulationApp` bootstraps the Kit extension
paths, so all USD imports happen below that line.

What it proves, end to end:
  1. A real Isaac Sim `SimulationApp` starts (headless).
  2. It opens the exact `.usda` the helios env wrote — no re-authoring.
  3. `pxr` independently traverses the stage, counting points/faces across every mesh
     (descending into instance proxies so an instanced orchard is counted in full) and
     resolves the bound UsdPreviewSurface materials + their UsdUVTexture asset paths, so we
     confirm the textures actually resolve on the consumer side — this is the texture-fix
     validation, not just geometry round-trip.
  4. A camera is framed on the whole scene and the viewport is rendered to PNG, so there is
     a human-checkable artifact that the trees loaded, are lit, and are textured.
"""

import argparse
import json
import os

# --- Bootstrap Isaac Sim BEFORE any pxr / omni import -------------------------------
from isaacsim import SimulationApp  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(__file__)
    ap.add_argument("--usd", default=os.path.join(here, "output", "apple_tree.usda"))
    ap.add_argument(
        "--report", default=os.path.join(here, "output", "apple_tree_export_report.json")
    )
    ap.add_argument(
        "--screenshot", default=os.path.join(here, "output", "isaacsim_apple_tree.png")
    )
    ap.add_argument("--render-frames", type=int, default=90)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    # Camera framing as multiples of the scene extent (tweak for hero shots).
    ap.add_argument("--cam-x", type=float, default=0.9)
    ap.add_argument("--cam-y", type=float, default=-1.1)
    ap.add_argument("--cam-z", type=float, default=0.45)
    return ap.parse_args()


def main():
    args = parse_args()
    result = {"usd": args.usd, "opened": False, "roundtrip_ok": None, "errors": []}

    sim_app = SimulationApp({"headless": True, "width": args.width, "height": args.height})

    try:
        import omni.usd
        from pxr import Usd, UsdGeom, UsdShade, UsdLux, Sdf, Gf
        import omni.replicator.core as rep

        # 1. Open the exact exported stage.
        ok = omni.usd.get_context().open_stage(args.usd)
        result["opened"] = bool(ok)
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Stage failed to open")

        result["up_axis"] = str(UsdGeom.GetStageUpAxis(stage))
        result["meters_per_unit"] = UsdGeom.GetStageMetersPerUnit(stage)

        # The exported stage is geometry-only (no lights) -> RTX renders it black.
        dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
        dome.CreateIntensityAttr(1200.0)
        distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/SunLight"))
        distant.CreateIntensityAttr(3000.0)
        distant.CreateAngleAttr(0.53)
        UsdGeom.Xformable(distant.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 25.0))
        result["added_lights"] = ["/World/DomeLight", "/World/SunLight"]

        # 2. Traverse every mesh (incl. instance proxies) and sum points/faces.
        trav = Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
        n_points = n_faces = n_meshes = 0
        materials_seen = set()
        texture_assets = set()
        for prim in trav:
            if prim.IsA(UsdGeom.Mesh):
                m = UsdGeom.Mesh(prim)
                pts = m.GetPointsAttr().Get()
                fvc = m.GetFaceVertexCountsAttr().Get()
                n_points += len(pts) if pts else 0
                n_faces += len(fvc) if fvc else 0
                n_meshes += 1
            if prim.IsA(UsdShade.Material):
                materials_seen.add(str(prim.GetPath()))
            if prim.IsA(UsdShade.Shader):
                sh = UsdShade.Shader(prim)
                sid = sh.GetIdAttr().Get()
                if sid == "UsdUVTexture":
                    fp = sh.GetInput("file")
                    if fp:
                        val = fp.Get()
                        if val:
                            texture_assets.add(str(val).lstrip("@").rstrip("@"))

        result["pxr_num_points"] = n_points
        result["pxr_num_faces"] = n_faces
        result["pxr_num_meshes"] = n_meshes
        result["materials"] = sorted(materials_seen)
        # Resolve texture asset paths against the stage and confirm they exist on disk.
        layer_dir = os.path.dirname(args.usd)
        tex_report = []
        for t in sorted(texture_assets):
            resolved = t if os.path.isabs(t) else os.path.normpath(os.path.join(layer_dir, t))
            tex_report.append({"path": t, "exists": os.path.exists(resolved)})
        result["textures"] = tex_report
        result["textures_all_exist"] = all(x["exists"] for x in tex_report) if tex_report else False

        if os.path.exists(args.report):
            with open(args.report) as f:
                rep_json = json.load(f)
            exp_p = rep_json.get("num_points")
            exp_f = rep_json.get("num_faces")
            n_inst = rep_json.get("n_instances", 1)
            # Orchard reports per-unique-tree points; multiply for instanced total when given.
            if "total_num_points" in rep_json:
                exp_p = rep_json["total_num_points"]
                exp_f = rep_json["total_num_faces"]
            result["export_num_points"] = exp_p
            result["export_num_faces"] = exp_f
            result["roundtrip_ok"] = (exp_p == n_points and exp_f == n_faces)

        # 3. Frame a camera on the whole scene bbox and render to PNG via Replicator.
        # Bound the whole scene from the pseudo-root: the geometry lives under the file's
        # defaultPrim (/AppleTree or /Orchard), not under the /World scope we add lights to.
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_],
                                       useExtentsHint=False)
        rng = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
        cmin, cmax = rng.GetMin(), rng.GetMax()
        center = [(cmin[i] + cmax[i]) * 0.5 for i in range(3)]
        extent = max((cmax[i] - cmin[i]) for i in range(3)) or 1.0

        cam = rep.create.camera(
            position=(center[0] + args.cam_x * extent, center[1] + args.cam_y * extent,
                      center[2] + args.cam_z * extent),
            look_at=tuple(center),
        )
        rp = rep.create.render_product(cam, (args.width, args.height))
        writer = rep.WriterRegistry.get("BasicWriter")
        out_dir = os.path.abspath(os.path.dirname(args.screenshot))  # BasicWriter needs abs
        writer.initialize(output_dir=out_dir, rgb=True)
        writer.attach([rp])

        for _ in range(args.render_frames):
            sim_app.update()
        for _ in range(3):
            rep.orchestrator.step()
            for _ in range(10):
                sim_app.update()

        result["render_dir"] = out_dir
        result["screenshot_hint"] = "BasicWriter wrote rgb_*.png in render_dir"

    except Exception as e:  # noqa: BLE001
        result["errors"].append(repr(e))
    finally:
        out_json = os.path.splitext(args.screenshot)[0] + "_open_report.json"
        try:
            with open(out_json, "w") as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass
        print(json.dumps(result, indent=2))
        sim_app.close()


if __name__ == "__main__":
    main()
