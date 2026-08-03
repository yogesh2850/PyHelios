# Cosmos-Transfer synthetic data from the phase9 orchard

Isaac Sim capture of Cosmos-Transfer-ready clips from
`yogesh_dev/phase9/output/orchard_scene.usda` (45-tree orchard, soil ground,
textured sky dome, distant hills, per-part semantic labels). Data collection
only — the Cosmos model itself is NOT downloaded or run here.

- Capture script: `cosmos_capture.py` (run with `/home/yogesh/isaacsim/python.sh`)
- Pipeline: `run_cosmos_capture.sh` (waits for the isaac_r2 GPU job, smoke
  test, full capture, output verification); watcher: `watch_cosmos.sh`
- Output: `yogesh_dev/world_model/output/cosmos_isaac/`
  (smoke clip: `.../cosmos_isaac_smoke/`)

## Camera: ZED X Mini vs Cosmos 720p reconciliation

The robot camera is a ZED X Mini with the 2.2 mm wide lens: native
**1920x1200 (16:10)**, ~**110° horizontal FOV**. Cosmos-Transfer expects
**1280x720 (16:9)** input video. We capture directly at Cosmos's 720p but
with the ZED's horizontal FOV, so horizontal scene geometry (what enters the
frame, perspective, apparent object sizes across the row) matches the real
camera; only the vertical FOV is slightly reduced by the 16:9 crop
(77.6° vs ~80° native). One mono view (Cosmos-Transfer takes a single view).

Exact intrinsics used (USD camera + implied pinhole at 1280x720):

| quantity | value |
|---|---|
| focal length | 2.2 mm |
| horizontal aperture | 6.2839 mm (= 2·2.2·tan 55°) |
| vertical aperture | 3.5347 mm |
| HFOV / VFOV | 110.0° / 77.6° |
| fx = fy | 448.1 px |
| cx, cy | 640, 360 |
| height / pitch | 1.2 m / −5° |
| clip fps | 24 (stage default timeCodesPerSecond) |

## Clips and trajectories

**14 clips x 121 frames** (the Cosmos-Transfer clip length) at 1280x720,
speeds 1.4–2.7 m/s. Trajectory diversity is the point:

- 6 passes per 6 m alley (y = −3 and y = +3): both directions, center and
  ±0.7–0.8 m lateral offsets, gentle S-curves (yaw swings ±40–50°), and
  yaw-scan passes (±0.3–0.35 rad sinusoidal scanning), plus small lateral
  sway / height bob / yaw noise on every pass.
- 2 headland turn-outs: full 180° arcs around each row end
  (x ≈ ±33, r = 3 m), connecting the two alleys.

Exact per-clip specs and achieved spans/speeds: `cosmos_isaac/manifest.json`.

## Output layout (CosmosWriter, omni.replicator.core 1.13.27)

```
cosmos_isaac/
  manifest.json                 # camera intrinsics, seg palette, clip specs
  clip_0000/ ... clip_0013/
    rgb/rgb_0000.png ... rgb_0120.png
    segmentation/segmentation_*.png     # colorized semantic segmentation
    shaded_seg/shaded_seg_*.png
    depth/depth_*.png                   # colorized distance_to_camera
    edges/edges_*.png                   # Canny on shaded seg
    rgb.mp4  segmentation.mp4  shaded_seg.mp4  depth.mp4  edges.mp4   # 24 fps, NVENC
```

Segmentation palette (`segmentation_mapping` passed to the writer):

| class | RGB |
|---|---|
| ground | 110, 70, 35 |
| fruit | 230, 30, 30 |
| leaf | 40, 170, 40 |
| shoot (trunk/branches) | 200, 140, 90 |
| hill | 120, 120, 160 |
| sky | 0, 0, 0 (unlabeled — the sky is a dome light with no geometry) |

Notes: the Fabric Scene Delegate is disabled during capture because it does
not resolve `SemanticsLabelsAPI` on GeomSubsets (the tree part labels);
"trunk" is the `shoot` class in the PyHelios asset taxonomy.

## Next step: running Cosmos-Transfer on this data (later, on a box with the weights)

1. Repo: `https://github.com/nvidia-cosmos/cosmos-transfer1`; weights:
   Hugging Face `nvidia/Cosmos-Transfer1-7B` (download via the repo's
   `checkpoints` scripts; ~40 GB+, needs an NGC/HF token).
2. Input format is exactly what this capture produces: 1280x720, 121-frame
   mp4s. Per clip, build the inference controlnet spec JSON:
   - `input_video_path`: `clip_NNNN/rgb.mp4`
   - control branches (weights are tunable per branch):
     `seg` → `clip_NNNN/segmentation.mp4`, `depth` → `clip_NNNN/depth.mp4`,
     `edge` → `clip_NNNN/edges.mp4` (or let it recompute canny), `vis` →
     blurred rgb (computed by the pipeline from the input video).
   - a text prompt describing the target photoreal orchard (season, lighting,
     cultivar, soil condition) — vary prompts per clip for appearance
     diversity at fixed geometry.
3. Run `transfer.py` (see the repo's `examples/inference_cosmos_transfer1_7b.md`);
   single-GPU inference fits in ~24 GB with offloading. Output is a
   photorealistic 121-frame video geometrically consistent with our
   simulation, i.e. paired (photoreal RGB, sim GT segmentation/depth/pose)
   training data for the world model.
