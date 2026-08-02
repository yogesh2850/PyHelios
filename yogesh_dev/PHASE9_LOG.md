# Phase 9 Log — PyHelios ↔ Isaac Sim bridge

Everything here is from real runs on this devbox (2026-08-02). Code in
`yogesh_dev/phase9/`, artifacts in `yogesh_dev/phase9/output/`.

## What was built

Three files, two conda envs, one USD file as the contract:

- `usd_export.py` — runs in `helios`. Builds a seeded apple tree via the existing
  `PlantArchitecture` path, pulls every primitive's real vertices/colours/normals out of the
  Context, and writes a valid ASCII `UsdGeomMesh` (`.usda`) directly. No `pxr` on this side
  (the `helios` env doesn't have it); a USDA file is plain text, so none is needed.
- `open_in_isaacsim.py` — runs in `env_isaaclab`. Boots a real Isaac Sim `SimulationApp`
  (headless), opens the exported stage, independently re-counts points/faces with `pxr`,
  checks stage up-axis/units, adds lighting, and renders a frame to PNG via Replicator.
- `run_phase9.sh` — orchestrates both halves; the `.usda` is the only thing passed between.

## T9.1 / T9.2 — Build + export (real numbers, `helios` env)

Single `apple` tree, `age=365 days`, `seed=9001` (a fresh seed band, disjoint from earlier
phases). Real output:

- **8,319 triangles**, **24,957 points** (non-indexed: each face owns its 3 points — simplest
  correct thing; Isaac de-dupes on import).
- Bounding box: **x∈[-0.796, 0.731] m, y∈[-0.728, 0.711] m, z∈[-0.0004, 2.033] m** — a
  ~1.5 m-wide, ~2.03 m-tall tree sitting on z≈0, exactly where Helios places it.
- Every primitive is a Helios triangle (type 1, 3 verts). The exporter also handles 4-vert
  patches (split on the 0-1-2 / 0-2-3 diagonal) generically, so larger/older trees that emit
  quads will export without code changes — none appeared at this age.
- Timing: build 0.27 s, geometry extract 0.04 s, USD write 0.01 s. File size **1.16 MB**.

## T9.3 — Open in a real Isaac Sim session (real, `env_isaaclab`)

`isaacsim.SimulationApp({headless:True})` → `omni.usd ... open_stage(apple_tree.usda)`.

- **Stage opened: yes.** Isaac Sim's own `pxr` traversal found the mesh at `/World/apple_tree`
  and counted **24,957 points / 8,319 faces** — an **exact match** to the producer-side export
  report (`roundtrip_ok = true`). This is a true round-trip: two different Python runtimes,
  two different USD implementations (our ASCII writer vs. `pxr`), same numbers.
- Stage metadata survived the trip: **up-axis `Z`, metersPerUnit `1.0`**.
- Kit cold-start ~47 s (first-launch shader warmup); the open + validate + render adds ~2 s
  on top. Total run ~49 s wall clock.
- Rendered frame: `output/rgb_0002.png` (1280×960). The apple tree is clearly visible —
  trunk, foliage, small fruit clusters.

## Gotchas found and worked around (real)

- **`pxr` is not importable in `env_isaaclab` until `SimulationApp` runs.** Isaac Sim's pip
  package only injects Kit's extension paths (where `pxr`/`omni.usd` live) after the app
  bootstraps. Fixed by importing `SimulationApp` first and doing every USD/omni import inside
  the app context, not at module top.
- **First render was pure black (23 KB PNG).** The exported stage is geometry-only, so RTX
  had nothing to light it with. Fixed by adding a `UsdLux.DomeLight` (ambient fill) + a
  `UsdLux.DistantLight` "sun" on the stage before rendering, and giving the renderer ~90
  warmup frames + 3 orchestrator steps so the denoiser converges. Lit frames are ~150 KB.
- **Texture-mapped primitives report colour `(0,0,0)`.** Helios leaves/bark are
  texture-mapped; `getPrimitiveColor` returns the solid base colour, which is black for those.
  Real split this run: **1,110 primitives carry a real Helios colour** (the apples) vs **7,209
  fall back**. To keep the opened stage legible rather than a black silhouette, black
  primitives get a per-organ tint from the surface normal (near-vertical faces → bark brown,
  else → leaf green). **Geometry (points/faces) is exact; only display colour is approximate.**
  Visible artifact: the trunk renders green because its side faces have near-horizontal
  normals, which land on the leaf side of the heuristic — cosmetic only. Texture-accurate
  export (baking Helios's leaf/bark textures into USD `UsdUVTexture` + UV primvars) is the
  obvious next step and is deliberately out of this minimal-bridge scope.

## How to reproduce

```
yogesh_dev/phase9/run_phase9.sh            # 1 tree, age 365, seed 9001
yogesh_dev/phase9/run_phase9.sh 3 365 9001 # a 3-tree row instead
```

Artifacts: `apple_tree.usda`, `apple_tree_export_report.json`,
`isaacsim_apple_tree_open_report.json`, `rgb_000{0,1,2}.png`.
