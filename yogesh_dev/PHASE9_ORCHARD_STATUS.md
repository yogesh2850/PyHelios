# Phase 9.5 — Texture-accurate bridge + 45-tree apple orchard (STATUS)

Follow-on to Phase 9. Two asks: (1) fix the "green trunk" cosmetic fallback so the USD
carries Helios's real bark/leaf/fruit textures, and (2) export a fully-grown 45-tree apple
orchard (15 trees/row × 3 rows) at real-world spacing, then open + render it in Isaac Sim.

Both done, run for real.

## 1. The texture fix (why the trunk was green, and how it's fixed)

Helios apple primitives are **texture-mapped**, so `getPrimitiveColor` returns (0,0,0) — the
old bridge saw black and substituted a normal-based leaf/bark tint. The real signal is in
`getPrimitiveTextureFile(uuid)` (which image) + `getPrimitiveTextureUV(uuid)` (per-vertex
UVs), both exposed by PyHelios. The exporter (`usd_export.py`) now:

- reads the texture file + per-vertex UVs for every primitive (batch APIs — a fully-grown
  tree is ~270k primitives);
- authors one `UsdPreviewSurface` per distinct texture, wired to the image through a
  `UsdUVTexture` + a `UsdPrimvarReader_float2` reading a per-vertex `primvars:st`
  (faceVarying) array;
- groups faces by texture into `GeomSubset`s (family `materialBind`) and binds each to its
  material;
- for `AppleLeaf.png` (which has an alpha channel), wires the texture alpha into
  `opacity` + `opacityThreshold = 0.5`, so leaves render as real leaf silhouettes, not
  green quads;
- leaves the ~1.1k genuinely untextured primitives (petioles/shoots) on solid displayColor;
- copies the three textures (`AppleBark.jpg`, `AppleLeaf.png`, `AppleFruit.jpg`) next to the
  stage and references them relatively, so the `.usda` is portable.

Still authored as plain ASCII from the `helios` env (no `pxr` on the producer side).

**Validated in Isaac Sim**: single fully-grown tree opens, all 3 materials + textures
resolve on the `pxr` side (`textures_all_exist = true`), and the render shows textured green
foliage with alpha-cut leaves, real bark, and red apples — the black/green-trunk artifact is
gone. (`output/isaacsim_test_tree.png` / hero: `output/rgb_*.png`.)

## 2. The 45-tree orchard

`orchard_export.py` — fully-grown trees, real orchard geometry, **instanced**:

| Property | Value |
|---|---|
| Trees | 45 (3 rows × 15) |
| Growth phase | fully grown, age 1825 d (~4.4 m tall, ~3.5 m canopy) |
| In-row spacing | 4.5 m (semi-dwarf/standard apple) |
| Between-row spacing | 6.0 m (tractor alley) |
| Block footprint | 63 m rows × 12 m across |
| Unique prototypes | 3 distinct seeded trees (9001/9002/9003), each rotated per placement |
| Geometry described | 47,773,215 points / 15,924,405 faces (all 45 instances) |
| `orchard.usda` size | ~14 KB (instanced references) |
| Unique tree geometry on disk | ~200 MB across 3 `trees/tree_0X.usda` |

Why instancing: 45 fully-grown trees as raw meshes would be a multi-GB file. Real orchards
are one cultivar of near-identical trees, so each position is an `instanceable = true`
reference to one of 3 prototypes with a per-tree translate + heading rotation. File size and
GPU memory scale with the 3 unique trees, not the 45 placements — but the stage still
describes all 45 in full.

**Validated in Isaac Sim**: `roundtrip_ok = true` — `pxr` traversed the stage with
`Usd.TraverseInstanceProxies()` and counted the exact 47,773,215 pts / 15,924,405 faces
across 45 meshes; up-axis `Z`, metersPerUnit `1.0`, all textures resolve; orchard rendered
(`output/isaacsim_orchard*.png`).

## Run it

```
# single fully-grown textured tree (default age now 1825 d):
yogesh_dev/phase9/run_phase9.sh 1 1825 9001

# the 45-tree orchard (rows per_row age in_row row_spacing n_unique):
yogesh_dev/phase9/run_orchard.sh 3 15 1825 4.5 6.0 3
```

## Honest limitations

- Uses **displayColor-neutral (white) base under the texture**; Helios's per-primitive
  texture-color overrides / masks are not applied, so a leaf's exact per-instance tint is
  the texture's own colour, not any Helios recolour. Visually correct for apple; a
  `resolveMaterialTextures`-aware path would be the fully-faithful version.
- Lights are still added on the Isaac side (Helios radiation sources ≠ `UsdLux`).
- 3 unique prototypes (not 45 unique trees) — a deliberate instancing choice for file
  size/GPU. Bump `--n-unique` for more variety at linear cost.

STATUS: DONE
