# Phase 9 Status — PyHelios ↔ Isaac Sim bridge

All three subtasks implemented and executed end-to-end with real data. Code in
`yogesh_dev/phase9/`; artifacts in `yogesh_dev/phase9/output/`. Full detail (real numbers,
gotchas found + fixed) in `PHASE9_LOG.md`.

| Subtask | What it does | Result |
|---|---|---|
| T9.1 | Define a tree in PyHelios (seeded `apple`, age 365, seed 9001) | 8,319 tris / 24,957 pts, bbox ~1.5×1.5×2.03 m |
| T9.2 | Save as OpenUSD (`.usda`, `UsdGeomMesh`, Z-up, metres) | 1.16 MB, valid, geometry-only |
| T9.3 | Open in a real Isaac Sim 5.1 session + render | opened=true, pxr round-trip exact, tree rendered to PNG |

## Key real results

- **Round-trip is exact**: our ASCII USD writer (helios env, no pxr) → Isaac Sim's `pxr`
  reader counted the same 24,957 points / 8,319 faces. Up-axis `Z` and metersPerUnit `1.0`
  preserved. `roundtrip_ok = true`.
- **Real Isaac Sim session, real render**: `SimulationApp` headless, ~47 s Kit cold start,
  stage opened, frame rendered (`output/rgb_0002.png`) with the apple tree visibly loaded.
- **Two envs, one contract**: producer `helios` (no pxr, doesn't need it), consumer
  `env_isaaclab` (Isaac Sim 5.1.0). The `.usda` is the only thing crossing between them.

## Honest limitations (logged, not hidden)

- **Display colour is approximate for texture-mapped primitives** (7,209 of 8,319 fall back to
  a normal-based leaf/bark tint; the 1,110 apples keep their real Helios colour). Geometry is
  exact. The trunk rendering green is a cosmetic fallback artifact, not a geometry error.
- **No textures / materials / UVs exported yet** — this is a geometry+displayColor bridge.
  Baking Helios leaf/bark textures into USD `UsdUVTexture` + UV primvars is the clear next
  step, deliberately out of this minimal-bridge scope.
- **Lights are added on the Isaac side**, not exported from Helios (Helios's radiation light
  sources don't map 1:1 to `UsdLux`). Fine for "open and look at the tree"; a physically
  matched lighting export would be future work.

STATUS: DONE
