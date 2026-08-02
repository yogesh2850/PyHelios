"""Isaac world-model T1 -- scene realism for the 45-tree orchard.

Reads yogesh_dev/phase9/output/orchard.usda (NEVER modified) and writes a new
orchard_scene.usda next to it.

MEASURED CONSTRAINT that shapes this file: Isaac Sim 6.0.1's semantic
segmentation does NOT resolve SemanticsLabelsAPI authored on GeomSubsets --
a minimal directly-authored labeled subset renders as UNLABELLED (with the
Fabric Scene Delegate on OR off; see debug_subset.py), while labels on whole
prims (ground, hills) resolve fine. So each of the 3 prototype trees is SPLIT
into three per-material meshes (bark / leaf / fruit) with the label on the
mesh prim itself, written as trees/tree_XX_sem.usd (binary; the original
tree_XX.usda files are untouched). The split preserves points, faceVarying
UVs, uniform displayColor and material bindings exactly -- faces are only
regrouped, never altered.

The composed scene:

  /World/Orchard          45 tree Xforms (transforms copied from orchard.usda)
                          each referencing a split prototype, NOT instanced so
                          every labeled mesh is an ordinary prim
  /World/Ground           800 x 800 m soil quad at z=0, tiled procedural soil
                          texture (generated locally -- no Nucleus / web assets)
  /World/Hills            two rings of squashed-ellipsoid hills at ~350/470 m so
                          the horizon is not empty, forest-noise texture
  /World/SkyLight         DomeLight with a generated lat-long sky texture
                          (gradient + clouds). The sky is *light*, not geometry:
                          at capture time no-hit pixels (depth = inf) become
                          semantic class 6 with the dataset's -1.0 depth
                          sentinel, exactly like the Helios dataset stores sky.
  /World/Sun              DistantLight, same intensity/orientation family as the
                          Phase 9 hero render.

Semantic labels use UsdSemantics.LabelsAPI instance "class" (the scheme Isaac
Sim 6.0 / omni.replicator 1.13 reads):
  subset_AppleFruit -> "fruit"   subset_AppleLeaf -> "leaf"
  subset_AppleBark  -> "shoot"   Ground -> "ground"   Hills -> "hill"

Runs with the Isaac Sim interpreter (/home/yogesh/isaacsim/python.sh) but only
needs pxr + numpy + PIL -- no SimulationApp, so it is fast.
"""

import json
import os
import sys

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE9_OUT = os.path.normpath(os.path.join(HERE, "..", "phase9", "output"))
ORCHARD = os.path.join(PHASE9_OUT, "orchard.usda")
SCENE = os.path.join(PHASE9_OUT, "orchard_scene.usda")
TEXDIR = os.path.join(PHASE9_OUT, "textures_scene")
TREES_DIR = os.path.join(PHASE9_OUT, "trees")

SUBSET_LABEL = {
    "subset_AppleFruit": "fruit",
    "subset_AppleLeaf": "leaf",
    "subset_AppleBark": "shoot",
}


# ---------------------------------------------------------------------------
# procedural textures (deterministic, offline)
# ---------------------------------------------------------------------------
def _value_noise(rng, size, octaves=5, persistence=0.55):
    """Multi-octave bilinear value noise in [0,1], (size,size) float32."""
    out = np.zeros((size, size), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        cells = 4 * (2 ** o)
        grid = rng.random((cells + 1, cells + 1)).astype(np.float32)
        # bilinear upsample grid -> size
        xs = np.linspace(0, cells, size, endpoint=False)
        i = xs.astype(int)
        f = (xs - i)[None, :]
        g = grid[np.ix_(i, i)]
        gx = grid[np.ix_(i, i + 1)]
        gy = grid[np.ix_(i + 1, i)]
        gxy = grid[np.ix_(i + 1, i + 1)]
        fy = f.T
        layer = (g * (1 - f) + gx * f) * (1 - fy) + (gy * (1 - f) + gxy * f) * fy
        out += amp * layer
        total += amp
        amp *= persistence
    return out / total


def make_soil_texture(path, size=1024, seed=7):
    rng = np.random.default_rng(seed)
    n = _value_noise(rng, size, octaves=6)
    fine = _value_noise(rng, size, octaves=8, persistence=0.7)
    base = np.array([0.38, 0.27, 0.17])          # loamy brown
    dark = np.array([0.24, 0.16, 0.10])          # damp patches
    light = np.array([0.52, 0.40, 0.27])         # dry crumbs
    t = n[..., None]
    rgb = dark * (1 - t) + light * t
    rgb = 0.75 * rgb + 0.25 * base
    rgb += (fine[..., None] - 0.5) * 0.10        # grain
    # sparse pebbles / clods
    speck = (fine > 0.82)[..., None]
    rgb = np.where(speck, rgb * 0.6, rgb)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)


def make_hills_texture(path, size=512, seed=11):
    rng = np.random.default_rng(seed)
    n = _value_noise(rng, size, octaves=7, persistence=0.6)
    dark = np.array([0.05, 0.13, 0.05])
    mid = np.array([0.10, 0.24, 0.09])
    light = np.array([0.20, 0.34, 0.13])
    t = n[..., None]
    rgb = np.where(t < 0.5, dark * (1 - 2 * t) + mid * (2 * t),
                   mid * (2 - 2 * t) + light * (2 * t - 1))
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)


def make_sky_texture(path, w=2048, h=1024, seed=3):
    """Lat-long sky for the DomeLight: blue gradient + soft cumulus band."""
    rng = np.random.default_rng(seed)
    v = np.linspace(0, 1, h)[:, None]            # 0 = zenith, 1 = nadir
    zen = np.array([0.16, 0.34, 0.72])
    hor = np.array([0.72, 0.82, 0.92])
    gnd = np.array([0.45, 0.42, 0.38])           # below horizon (rarely visible)
    up = np.clip(v / 0.5, 0, 1)[..., None]       # zenith->horizon ramp
    sky = zen * (1 - up) + hor * up
    rgb = np.repeat(sky, w, axis=1).reshape(h, w, 3)
    rgb[v[:, 0] > 0.55] = gnd
    # clouds: noise stretched horizontally, only in a band above the horizon
    n = _value_noise(rng, 1024, octaves=6, persistence=0.6)
    n = np.asarray(Image.fromarray((n * 255).astype(np.uint8)).resize((w, h)),
                   np.float32) / 255.0
    band = np.exp(-((v - 0.33) / 0.13) ** 2)     # cloud band around mid-sky
    cloud = np.clip((n - 0.58) * 4.0, 0, 1) * band
    rgb = rgb * (1 - cloud[..., None]) + np.array([0.97, 0.97, 0.98]) * cloud[..., None]
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)


# ---------------------------------------------------------------------------
# USD authoring helpers
# ---------------------------------------------------------------------------
def tag(prim, label):
    """UsdSemantics.LabelsAPI:class = [label] (authored generically so it works
    even if the UsdSemantics python module is absent)."""
    prim.AddAppliedSchema("SemanticsLabelsAPI:class")
    attr = prim.CreateAttribute("semantics:labels:class", Sdf.ValueTypeNames.TokenArray)
    attr.Set([label])


def make_textured_material(stage, path, tex_rel, uv_scale=1.0, roughness=0.9,
                           emissive=False):
    mat = UsdShade.Material.Define(stage, path)
    surf = UsdShade.Shader.Define(stage, path + "/Surface")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    st = UsdShade.Shader.Define(stage, path + "/stReader")
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    tex = UsdShade.Shader.Define(stage, path + "/Tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_rel)
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    if uv_scale != 1.0:
        xf = UsdShade.Shader.Define(stage, path + "/uvScale")
        xf.CreateIdAttr("UsdTransform2d")
        xf.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(
            st.ConnectableAPI(), "result")
        xf.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(uv_scale, uv_scale))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            xf.ConnectableAPI(), "result")
    else:
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            st.ConnectableAPI(), "result")

    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), "rgb")
    if emissive:
        surf.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
    return mat


def split_tree_prototype(src_path, dst_path):
    """tree_XX.usda (one mesh + 3 material GeomSubsets) -> tree_XX_sem.usd
    (three meshes, one per material, labeled at prim level)."""
    sstage = Usd.Stage.Open(src_path)
    mesh = UsdGeom.Mesh(sstage.GetPrimAtPath("/AppleTree/Geom"))
    pts = np.asarray(mesh.GetPointsAttr().Get(), np.float32)
    fvi = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.int64)
    fvc = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), np.int64)
    assert (fvc == 3).all(), "splitter assumes an all-triangle mesh"
    pv_api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    st = np.asarray(pv_api.GetPrimvar("st").Get(), np.float32)
    dc_pv = pv_api.GetPrimvar("displayColor")
    dc = np.asarray(dc_pv.Get(), np.float32) if dc_pv and dc_pv.Get() is not None else None
    normals = mesh.GetNormalsAttr().Get()
    normals = (np.asarray(normals, np.float32)
               if normals is not None and len(normals) == len(fvi) else None)
    double_sided = bool(mesh.GetDoubleSidedAttr().Get())

    if os.path.exists(dst_path):
        os.remove(dst_path)
    dstage = Usd.Stage.CreateNew(dst_path)
    UsdGeom.SetStageUpAxis(dstage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(dstage, 1.0)
    root = UsdGeom.Xform.Define(dstage, "/AppleTree")
    dstage.SetDefaultPrim(root.GetPrim())
    # copy the Looks scope verbatim; both layers live in trees/, so the
    # relative ./textures/... asset paths keep resolving
    Sdf.CreatePrimInLayer(dstage.GetRootLayer(), "/AppleTree/Looks")
    Sdf.CopySpec(sstage.GetRootLayer(), Sdf.Path("/AppleTree/Looks"),
                 dstage.GetRootLayer(), Sdf.Path("/AppleTree/Looks"))

    for sub in mesh.GetPrim().GetChildren():
        label = SUBSET_LABEL.get(sub.GetName())
        if not label:
            continue
        faces = np.asarray(UsdGeom.Subset(sub).GetIndicesAttr().Get(), np.int64)
        idx = (faces[:, None] * 3 + np.arange(3)[None, :]).reshape(-1)
        uniq, inv = np.unique(fvi[idx], return_inverse=True)
        mat_name = sub.GetName().replace("subset_", "")
        m = UsdGeom.Mesh.Define(dstage, f"/AppleTree/Geom_{mat_name}")
        sub_pts = pts[uniq]
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(sub_pts)))
        m.CreateFaceVertexCountsAttr(
            Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
        m.CreateFaceVertexIndicesAttr(
            Vt.IntArray.FromNumpy(np.ascontiguousarray(inv.astype(np.int32))))
        m.CreateExtentAttr([Gf.Vec3f(*[float(v) for v in sub_pts.min(0)]),
                            Gf.Vec3f(*[float(v) for v in sub_pts.max(0)])])
        m.CreateDoubleSidedAttr(double_sided)
        m.CreateSubdivisionSchemeAttr("none")
        stpv = UsdGeom.PrimvarsAPI(m.GetPrim()).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        stpv.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(st[idx])))
        if dc is not None and len(dc) == len(fvc):
            dcpv = UsdGeom.PrimvarsAPI(m.GetPrim()).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray,
                UsdGeom.Tokens.uniform)
            dcpv.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(dc[faces])))
        if normals is not None:
            m.CreateNormalsAttr(
                Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(normals[idx])))
            m.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(
            UsdShade.Material(dstage.GetPrimAtPath(f"/AppleTree/Looks/{mat_name}")))
        tag(m.GetPrim(), label)
    dstage.GetRootLayer().Save()


def assemble_orchard(stage):
    """Recreate the 45-tree grid from orchard.usda's layer specs, pointing each
    tree at its split prototype. Returns (n_trees, errors)."""
    lay = Sdf.Layer.FindOrOpen(ORCHARD)
    orch_spec = lay.GetPrimAtPath("/Orchard")
    UsdGeom.Xform.Define(stage, "/World/Orchard")
    n, errors = 0, []
    for spec in orch_spec.nameChildren:
        if not spec.name.startswith("Tree_"):
            continue
        items = spec.referenceList.GetAddedOrExplicitItems()
        if not items:
            errors.append(f"{spec.name}: no reference")
            continue
        asset = items[0].assetPath.replace(".usda", "_sem.usd")
        t = spec.attributes["xformOp:translate"].default
        r = spec.attributes["xformOp:rotateZ"].default
        prim = stage.DefinePrim(f"/World/Orchard/{spec.name}")
        prim.GetReferences().AddReference(asset)
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(t))
        xf.AddRotateZOp().Set(float(r))
        n += 1
    return n, errors


def add_ground(stage, tex_rel):
    mesh = UsdGeom.Mesh.Define(stage, "/World/Ground")
    s = 400.0
    mesh.CreatePointsAttr([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateExtentAttr([(-s, -s, 0), (s, s, 0)])
    mesh.CreateDoubleSidedAttr(True)
    pv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    reps = 100.0  # 8 m per soil tile
    pv.Set([(0, 0), (reps, 0), (reps, reps), (0, reps)])
    mat = make_textured_material(stage, "/World/Looks/Soil", tex_rel, roughness=0.95)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    tag(mesh.GetPrim(), "ground")
    return mesh


def add_hills(stage, tex_rel):
    rng = np.random.default_rng(23)
    mat = make_textured_material(stage, "/World/Looks/Hills", tex_rel, uv_scale=6.0,
                                 roughness=1.0)
    UsdGeom.Xform.Define(stage, "/World/Hills")
    k = 0
    for ring_r, n, rz_lo, rz_hi in ((350.0, 16, 22, 45), (470.0, 12, 40, 75)):
        for i in range(n):
            ang = 2 * np.pi * (i + rng.uniform(-0.2, 0.2)) / n
            r = ring_r * rng.uniform(0.92, 1.08)
            cx, cy = r * np.cos(ang), r * np.sin(ang)
            sph = UsdGeom.Sphere.Define(stage, f"/World/Hills/Hill_{k:02d}")
            sph.CreateRadiusAttr(1.0)
            sph.CreateExtentAttr([(-1, -1, -1), (1, 1, 1)])
            xf = UsdGeom.Xformable(sph.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(float(cx), float(cy), 0.0))
            sx = rng.uniform(90, 170)
            sy = rng.uniform(90, 170)
            sz = rng.uniform(rz_lo, rz_hi)
            xf.AddScaleOp().Set(Gf.Vec3f(float(sx), float(sy), float(sz)))
            UsdShade.MaterialBindingAPI.Apply(sph.GetPrim()).Bind(mat)
            tag(sph.GetPrim(), "hill")
            k += 1
    return k


def add_lights(stage, sky_rel):
    dome = UsdLux.DomeLight.Define(stage, "/World/SkyLight")
    dome.CreateIntensityAttr(1000.0)
    dome.CreateTextureFileAttr(sky_rel)
    dome.CreateTextureFormatAttr("latlong")
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(3000.0)
    sun.CreateAngleAttr(0.53)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 25.0))


def main():
    assert os.path.isfile(ORCHARD), f"missing {ORCHARD}"
    os.makedirs(TEXDIR, exist_ok=True)
    report = {"scene": SCENE, "errors": []}

    print("generating textures ...", flush=True)
    make_soil_texture(os.path.join(TEXDIR, "soil.png"))
    make_hills_texture(os.path.join(TEXDIR, "hills.png"))
    make_sky_texture(os.path.join(TEXDIR, "sky_latlong.png"))

    print("splitting tree prototypes by material subset ...", flush=True)
    for i in range(3):
        src = os.path.join(TREES_DIR, f"tree_{i:02d}.usda")
        dst = os.path.join(TREES_DIR, f"tree_{i:02d}_sem.usd")
        split_tree_prototype(src, dst)
        print(f"  {os.path.basename(dst)}: {os.path.getsize(dst) / 1e6:.1f} MB",
              flush=True)

    print("composing stage ...", flush=True)
    if os.path.exists(SCENE):
        os.remove(SCENE)
    stage = Usd.Stage.CreateNew(SCENE)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    n_trees, errs = assemble_orchard(stage)
    report["errors"] += errs
    report["n_trees"] = n_trees
    if n_trees != 45:
        report["errors"].append(f"expected 45 trees, got {n_trees}")

    add_ground(stage, "./textures_scene/soil.png")
    report["n_hills"] = add_hills(stage, "./textures_scene/hills.png")
    add_lights(stage, "./textures_scene/sky_latlong.png")

    stage.GetRootLayer().Save()
    print(f"wrote {SCENE}", flush=True)

    # -- verification pass on a fresh open ---------------------------------
    vstage = Usd.Stage.Open(SCENE)
    labels = {}
    trav = Usd.PrimRange.Stage(vstage, Usd.TraverseInstanceProxies())
    for prim in trav:
        for schema in prim.GetAppliedSchemas():
            if schema.startswith("SemanticsLabelsAPI:"):
                val = prim.GetAttribute("semantics:labels:class").Get()
                if val:
                    for v in val:
                        labels[str(v)] = labels.get(str(v), 0) + 1
    report["label_counts"] = labels
    need = {"fruit": 45, "leaf": 45, "shoot": 45, "ground": 1}
    for k, n in need.items():
        if labels.get(k, 0) < n:
            report["errors"].append(f"label '{k}': {labels.get(k, 0)} < {n}")

    with open(os.path.join(PHASE9_OUT, "orchard_scene_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1), flush=True)
    if report["errors"]:
        print("SCENE_FAIL", flush=True)
        sys.exit(1)
    print("SCENE_OK", flush=True)


if __name__ == "__main__":
    main()
