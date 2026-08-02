# Task: Phase 9 — PyHelios ↔ Isaac Sim bridge

New phase, added on top of the completed Phases 0-8. Goal: stand up the smallest real,
end-to-end bridge between the Helios/PyHelios world and NVIDIA Isaac Sim, so a tree that the
earlier phases reason about can be handed to Isaac Sim as native geometry.

## Scope (deliberately minimal, but fully real)

1. **T9.1 — Define a tree in PyHelios.** Reuse the existing `apple` library model
   (`apple_tree.py` build path), seeded for determinism (Phase 2/5/8 finding).
2. **T9.2 — Save it as USD.** Export the tree's real primitive geometry to an OpenUSD
   stage (`.usda`). Must be a valid `UsdGeomMesh` with correct up-axis and units.
3. **T9.3 — Open it in an Isaac Sim session.** Launch a real Isaac Sim `SimulationApp`,
   open the exported stage, validate it independently with `pxr`, and render a frame to a
   PNG as human-checkable proof the tree loaded.

## Environment (verified, don't re-check)

- Producer: `helios` conda env (`~/anaconda3/envs/helios/bin/python`, py3.12). Has
  `pyhelios`; **no `pxr`** — and doesn't need it (USDA is ASCII).
- Consumer: `env_isaaclab` conda env (`~/anaconda3/envs/env_isaaclab/bin/python`, py3.11)
  with Isaac Sim 5.1.0 (pip package) + IsaacLab. `pxr`/Kit are only importable after
  `isaacsim.SimulationApp` bootstraps.
- Helios is Z-up, metres → maps 1:1 onto USD `upAxis="Z"`, `metersPerUnit=1`. No transform.

## HARD CONSTRAINT

Everything lives under `/home/yogesh/PyHelios/yogesh_dev/phase9/`. Do not edit
`apple_tree.py` or any other top-level script. Keep the two-env split — the `.usda` file is
the only contract between producer and consumer.

## Honesty requirements (same standard as Phases 0-8)

- Report **real** exported counts, real bbox, real Isaac Sim round-trip numbers, real
  timings. No mock stage, no faked screenshot.
- Any approximation (e.g. display colour for texture-mapped primitives) must be labelled as
  such in the log, with the geometry kept exact.
