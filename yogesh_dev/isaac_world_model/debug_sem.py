"""One-frame semantic debug: dump raw idToLabels + id histogram."""
import json
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from capture import Rig, RENDER_RES, SCENE

rig = Rig(SCENE, RENDER_RES, 16)
try:
    rig.warmup(6)
    rig.set_pose(np.array([2.2, -3.0, 1.2, math.pi / 2]))
    rig.step()
    raw = rig.ann["sem"].get_data()
    data, info = (raw.get("data"), raw.get("info", {})) if isinstance(raw, dict) else (raw, {})
    data = np.asarray(data)
    ids, counts = np.unique(data, return_counts=True)
    print("SEM dtype/shape:", data.dtype, data.shape)
    print("SEM id histogram:", dict(zip(ids.tolist(), counts.tolist())))
    print("SEM info keys:", list(info.keys()))
    print("SEM idToLabels RAW:", json.dumps(info.get("idToLabels", {}), default=str, indent=1))
    rawi = rig.ann["inst"].get_data()
    di, ii = (rawi.get("data"), rawi.get("info", {})) if isinstance(rawi, dict) else (rawi, {})
    di = np.asarray(di)
    print("INST dtype/shape:", di.dtype, di.shape, "n_unique:", len(np.unique(di)))
    print("INST info keys:", list(ii.keys()))
    itl = ii.get("idToLabels", {})
    print("INST idToLabels (first 10):", json.dumps(dict(list(itl.items())[:10]), default=str))
finally:
    rig.close()
