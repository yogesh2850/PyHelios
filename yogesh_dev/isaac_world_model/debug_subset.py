"""Isolate why GeomSubset semantics fail for the referenced trees.

Builds a minimal stage with:
  A. a 2-triangle quad whose two GeomSubsets are labeled leaf / fruit (direct)
  B. ground quad labeled ground (direct)
  C. one referenced tree (de-instanced over + subset label overs, exactly like
     build_scene.py does)
renders one frame and dumps idToLabels.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE9_OUT = os.path.normpath(os.path.join(HERE, "..", "phase9", "output"))
MINI = os.path.join(PHASE9_OUT, "debug_subset_scene.usda")


def build():
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

    def tag(prim, label):
        prim.AddAppliedSchema("SemanticsLabelsAPI:class")
        prim.CreateAttribute("semantics:labels:class",
                             Sdf.ValueTypeNames.TokenArray).Set([label])

    if os.path.exists(MINI):
        os.remove(MINI)
    stage = Usd.Stage.CreateNew(MINI)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # A: quad of 2 triangles, subsets labeled leaf / fruit, upright facing +x
    m = UsdGeom.Mesh.Define(stage, "/World/TestQuad")
    m.CreatePointsAttr([(0, -1, 0), (0, 1, 0), (0, 1, 2), (0, -1, 2)])
    m.CreateFaceVertexCountsAttr([3, 3])
    m.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    m.CreateExtentAttr([(0, -1, 0), (0, 1, 2)])
    m.CreateDoubleSidedAttr(True)
    s1 = UsdGeom.Subset.Define(stage, "/World/TestQuad/leafpart")
    s1.CreateElementTypeAttr("face")
    s1.CreateFamilyNameAttr("materialBind")
    s1.CreateIndicesAttr([0])
    tag(s1.GetPrim(), "leaf")
    s2 = UsdGeom.Subset.Define(stage, "/World/TestQuad/fruitpart")
    s2.CreateElementTypeAttr("face")
    s2.CreateFamilyNameAttr("materialBind")
    s2.CreateIndicesAttr([1])
    tag(s2.GetPrim(), "fruit")

    # B: ground
    g = UsdGeom.Mesh.Define(stage, "/World/Gnd")
    g.CreatePointsAttr([(-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0)])
    g.CreateFaceVertexCountsAttr([4])
    g.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    g.CreateExtentAttr([(-50, -50, 0), (50, 50, 0)])
    tag(g.GetPrim(), "ground")

    # C: one referenced tree at y=3, overs like build_scene
    tree = stage.DefinePrim("/World/T0")
    tree.GetReferences().AddReference("./trees/tree_00.usda")
    tree.SetInstanceable(False)
    from pxr import UsdGeom as UG
    UG.Xformable(tree).AddTranslateOp().Set(Gf.Vec3d(3.0, 3.0, 0.0))
    geom = tree.GetChild("Geom")
    print("tree Geom children:", [c.GetName() for c in geom.GetChildren()])
    lab = {"subset_AppleFruit": "fruit", "subset_AppleLeaf": "leaf",
           "subset_AppleBark": "shoot"}
    for sub in geom.GetChildren():
        if sub.GetName() in lab:
            tag(sub, lab[sub.GetName()])

    dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
    dome.CreateIntensityAttr(1000.0)
    stage.GetRootLayer().Save()


def render():
    sys.path.insert(0, HERE)
    import numpy as np
    from capture import Rig, RENDER_RES
    rig = Rig(MINI, RENDER_RES, 16)
    try:
        rig.warmup(6)
        # camera at x=-4 looking +x: sees test quad at x=0 and tree at (3,3)
        rig.set_pose(np.array([-4.0, 1.0, 1.2, 0.0]))
        rig.step()
        raw = rig.ann["sem"].get_data()
        data, info = (raw.get("data"), raw.get("info", {})) if isinstance(raw, dict) else (raw, {})
        data = np.asarray(data)
        ids, counts = np.unique(data, return_counts=True)
        print("SEM id histogram:", dict(zip(ids.tolist(), counts.tolist())))
        print("SEM idToLabels RAW:",
              json.dumps(info.get("idToLabels", {}), default=str, indent=1))
    finally:
        rig.close()


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
    else:
        render()
