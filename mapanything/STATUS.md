# Status

MapAnything (facebookresearch/map-anything, real HuggingFace weights
`facebook/map-anything`) installed into a dedicated `mapanything` conda env
(python 3.12, torch 2.7.0+cu128) and run end to end against Phase 7's real
WAI dataset (`tree0_orbit16`, 16 views; `tree_t76_mvs_rig`, 42 views), both
`images_only` and `full` (+intrinsics+extrinsics+depth) conditioning
levels, on the RTX 5090. No auth wall, no incompatible-dependency
blocker. Real latency, real camera-pose accuracy, real landmark RMSE
(vs Phase 7 T7.2's conditioning axis), and real voxel-grid reconstruction
accuracy (vs Phase 7 T7.6/T7.7's classical baseline, same tree/ground
truth/grid) all measured and written to `LOG.md` and
`output/*.json`. See LOG.md's "Summary" section for the headline
findings.

STATUS: DONE
