### Camera-pose accuracy (all reconstruction models)

| model | scene | conditioning | given real poses? | frame policy | cam-centre RMSE direct (mm) | cam-centre RMSE aligned (mm) | recovered scale | rot err aligned (deg, mean) | s/view |
|---|---|---|---|---|---|---|---|---|---|
| da3_base | tree0_orbit16 | full | yes | input_frame_preserved | 0.0 | 0.0 | 1.000 | 0.00 | 0.0245 |
| da3_base | tree0_orbit16 | images_only | no | similarity_aligned | 7619.5 | 3674.1 | 16.020 | 76.44 | 0.0486 |
| da3_base | tree0_orbit16 | images_only_ray | no | similarity_aligned | 7626.4 | 3679.2 | 14.982 | 75.05 | 0.0474 |
| da3_base | tree_t76_mvs_rig | full | yes | input_frame_preserved | 0.0 | 0.0 | 1.000 | 0.00 | 0.0160 |
| da3_base | tree_t76_mvs_rig | images_only | no | similarity_aligned | 7888.0 | 4532.1 | 6.897 | 91.92 | 0.0181 |
| da3_base | tree_t76_mvs_rig | images_only_ray | no | similarity_aligned | 7680.5 | 4423.1 | 7.157 | 84.21 | 0.0217 |
| da3_large11 | tree0_orbit16 | full | yes | input_frame_preserved | 0.0 | 0.0 | 1.000 | 0.00 | 0.0304 |
| da3_large11 | tree0_orbit16 | images_only | no | similarity_aligned | 7898.8 | 2507.9 | 6.004 | 26.56 | 0.0538 |
| da3_large11 | tree0_orbit16 | images_only_ray | no | similarity_aligned | 7720.8 | 2502.5 | 6.307 | 26.17 | 0.0449 |
| da3_large11 | tree_t76_mvs_rig | full | yes | input_frame_preserved | 0.0 | 0.0 | 1.000 | 0.00 | 0.0294 |
| da3_large11 | tree_t76_mvs_rig | images_only | no | similarity_aligned | 6882.6 | 3173.0 | 6.640 | 48.33 | 0.0373 |
| da3_large11 | tree_t76_mvs_rig | images_only_ray | no | similarity_aligned | 6944.1 | 3203.3 | 6.896 | 43.47 | 0.0374 |
| pi3x | tree0_orbit16 | full | yes | similarity_aligned | 3614.4 | 257.2 | 1.985 | 1.96 | 0.0557 |
| pi3x | tree0_orbit16 | images_only | no | similarity_aligned | 4550.2 | 1053.4 | 2.271 | 10.37 | 0.0884 |
| pi3x | tree0_orbit16 | pose_intrinsics | yes | similarity_aligned | 3917.9 | 277.9 | 2.049 | 2.32 | 0.0505 |
| pi3x | tree_t76_mvs_rig | full | yes | similarity_aligned | 5244.0 | 1321.8 | 2.930 | 8.06 | 0.0770 |
| pi3x | tree_t76_mvs_rig | images_only | no | similarity_aligned | 6032.5 | 2969.9 | 3.270 | 51.85 | 0.0558 |
| pi3x | tree_t76_mvs_rig | pose_intrinsics | yes | similarity_aligned | 4217.8 | 499.0 | 2.166 | 3.10 | 0.0707 |
| vggt_1b | tree0_orbit16 | images_only | no | similarity_aligned | 6189.4 | 741.8 | 5.615 | 6.53 | 0.0574 |
| vggt_1b | tree0_orbit16 | images_only_laxconf | no | similarity_aligned | 6189.4 | 741.8 | 5.615 | 6.53 | 0.0547 |
| vggt_1b | tree_t76_mvs_rig | images_only | no | similarity_aligned | 6338.6 | 1557.1 | 5.160 | 6.45 | 0.0446 |
| vggt_1b | tree_t76_mvs_rig | images_only_laxconf | no | similarity_aligned | 6338.6 | 1557.2 | 5.160 | 6.46 | 0.0444 |
| mapanything | tree0_orbit16 | full | yes | camera0_anchored | 67.2 | 67.2 | 1.000 | 0.46 | 0.1247 |
| mapanything | tree0_orbit16 | images_only | no | similarity_aligned | n/a | 4640.9 | 8.750 | n/a | 0.1204 |
| mapanything | tree_t76_mvs_rig | full | yes | camera0_anchored | 87.8 | 87.8 | 1.000 | 0.46 | 0.1247 |
| mapanything | tree_t76_mvs_rig | images_only | no | similarity_aligned | n/a | 4554.6 | 2.749 | n/a | 0.1204 |

### Landmark-recovery RMSE, `tree_t76_mvs_rig`

| model | conditioning | given real poses? | landmark RMSE (mm) | median err (mm) | branch RMSE (mm) | fruit RMSE (mm) | n landmark-view obs | n landmarks seen |
|---|---|---|---|---|---|---|---|---|
| da3_base | full | yes | 377.2 | 232.6 | 377.3 | 372.9 | 50734 | 1210 |
| da3_base | images_only | no | 4775.7 | 4233.7 | 4787.5 | 4518.2 | 2637 | 769 |
| da3_base | images_only_ray | no | n/a | n/a | n/a | n/a | 0 | 0 |
| da3_large11 | full | yes | 398.4 | 233.7 | 398.5 | 395.9 | 50820 | 1210 |
| da3_large11 | images_only | no | 2849.6 | 2134.1 | 2847.6 | 2933.8 | 929 | 691 |
| da3_large11 | images_only_ray | no | 2112.9 | 2133.8 | 2114.3 | 2042.9 | 879 | 762 |
| pi3x | full | yes | 2990.8 | 2971.4 | 2989.8 | 3023.2 | 1395 | 1107 |
| pi3x | images_only | no | 5266.9 | 4576.5 | 5272.6 | 5096.6 | 1294 | 880 |
| pi3x | pose_intrinsics | yes | 1071.8 | 822.4 | 1073.3 | 1022.2 | 2289 | 1210 |
| vggt_1b | images_only | no | 381.0 | 245.2 | 381.9 | 301.1 | 816 | 816 |
| vggt_1b | images_only_laxconf | no | 451.6 | 253.8 | 454.8 | 313.8 | 1159 | 1146 |
| mapanything | full | yes | 398.8 | 232.7 | n/a | n/a | 47882 | 1210 |
| mapanything | images_only | no | 13872.0 | 13790.2 | n/a | n/a | 36261 | 1209 |

Reference numbers on the same axis:

```json
{
  "phase7_classical_proxy_t72_A_images_only_rmse_mm": 74.41804134049153,
  "phase7_classical_proxy_t72_C_intrinsics_extrinsics_rmse_mm": 4.645100644237657,
  "mapanything_images_only_rmse_mm": 13872.0,
  "mapanything_full_rmse_mm": 398.8,
  "mapanything_source": "yogesh_dev/mapanything_eval/output/landmark_rmse_report.json, produced by the same method on the same scene/landmarks -- quoted here so the running comparison table has one place to read from.",
  "caveat": "Phase 7's T7.2 proxy used its own 8-view/180deg-arc rig and a different real tree instance, and is sparse classical triangulation of exactly the queried landmarks (best case for classical geometry); every foundation model here is dense feed-forward prediction sampled at a landmark's projected pixel. Same conditioning axis and same metric, NOT a literal same-scene/same-paradigm rig."
}
```

### Voxel-grid reconstruction, `tree_t76_mvs_rig`

| model | conditioning | n occupied voxels (5mm) | recall <5mm | recall 5-10mm | recall 10-20mm | recall >20mm | fruit diam mean abs rel err | internode length rel err |
|---|---|---|---|---|---|---|---|---|
| da3_base | full | 1238877 | 0.939 | 0.797 | 0.848 | 0.863 | 0.2988 | 0.0584 |
| da3_base | images_only | 0 | 0.000 | 0.000 | 0.000 | 0.000 | n/a | n/a |
| da3_base | images_only_ray | 0 | 0.000 | 0.000 | 0.000 | 0.000 | n/a | n/a |
| da3_large11 | full | 1765992 | 0.848 | 0.861 | 0.930 | 0.740 | 0.2990 | 0.0656 |
| da3_large11 | images_only | 271026 | 0.121 | 0.045 | 0.156 | 0.075 | 0.2905 | -0.0664 |
| da3_large11 | images_only_ray | 203628 | 0.121 | 0.044 | 0.102 | 0.123 | 0.2877 | -0.0345 |
| pi3x | full | 35912 | 0.061 | 0.011 | 0.037 | 0.014 | 0.2729 | -0.6888 |
| pi3x | images_only | 0 | 0.000 | 0.000 | 0.000 | 0.000 | n/a | n/a |
| pi3x | pose_intrinsics | 849545 | 0.758 | 0.573 | 0.713 | 0.541 | 0.2971 | 0.0566 |
| vggt_1b | images_only | 419207 | 0.182 | 0.284 | 0.287 | 0.308 | 0.2846 | -0.0052 |
| vggt_1b | images_only_laxconf | 988137 | 0.515 | 0.523 | 0.611 | 0.637 | 0.2922 | 0.0372 |
| mapanything | full | 755220 | 1.000 | 0.853 | 0.877 | 0.959 | 0.2987 | 0.0536 |
| mapanything | images_only | 0 | 0.000 | 0.000 | 0.000 | 0.000 | n/a | n/a |

Phase 7 classical baseline:

```json
{
  "branch_recovery_rate_fine_grid_classical_t76": 0.5420560747663551,
  "fruit_recovery_rate_fine_grid_classical_t76": 1.0,
  "fruit_diameter_mean_abs_rel_err_classical_t77": 0.22450420505465235,
  "internode_length_rel_err_classical_t77": 0.11242804938623688,
  "caveat": "T7.6's classical baseline used real log-odds probabilistic depth fusion with free-space carving (phase4/occupancy_map.py); this script uses plain point-cloud voxelization of each model's mask-valid predicted points -- a simpler, weaker, and much denser fusion step on the model side, stated explicitly so a recall difference is not misread as purely a foundation-model difference."
}
```

### Per-view depth accuracy (depth-focused family)

| model | scene | prompt/conditioning | depth RMSE (mm) | MAE (mm) | AbsRel | delta1 | scale-aligned RMSE (mm) | median scale pred->gt | s/view |
|---|---|---|---|---|---|---|---|---|---|
| moge2_vitl | tree0_orbit16 | images_only | 14663.3 | 13959.8 | 2.7166 | 0.000 | 1234.2 | 0.2673 | 0.0596 |
| moge2_vitl | tree0_orbit16 | known_fov | 5750.8 | 5670.6 | 1.1083 | 0.000 | 561.7 | 0.4711 | 0.0449 |
| moge2_vitl | tree_t76_mvs_rig | images_only | 12527.8 | 11798.4 | 2.3446 | 0.000 | 1521.7 | 0.3311 | 0.0448 |
| moge2_vitl | tree_t76_mvs_rig | known_fov | 5276.1 | 5120.5 | 1.0260 | 0.000 | 771.1 | 0.5093 | 0.0448 |
| promptda_vitl | tree0_orbit16 | prompt_dense_noisy | 4836.8 | 4669.4 | 0.9041 | 0.062 | 117967.1 | 79.6240 | 0.0320 |
| promptda_vitl | tree0_orbit16 | prompt_sparse_clean | 4897.6 | 4755.3 | 0.9210 | 0.051 | 130918.1 | 100.2380 | 0.0318 |
| promptda_vitl | tree0_orbit16 | prompt_sparse_noisy | 4897.8 | 4754.8 | 0.9209 | 0.051 | 132723.4 | 101.9281 | 0.0459 |
| promptda_vitl | tree0_orbit16 | prompt_sparse_noisy_bgfill | 886.2 | 499.4 | 0.0970 | 0.893 | 842.5 | 0.9828 | 0.0538 |
| promptda_vitl | tree_t76_mvs_rig | prompt_dense_noisy | 4737.3 | 4573.9 | 0.9022 | 0.059 | 96510.5 | 65.8848 | 0.0321 |
| promptda_vitl | tree_t76_mvs_rig | prompt_sparse_clean | 4770.3 | 4615.6 | 0.9108 | 0.056 | 106162.6 | 78.5023 | 0.0319 |
| promptda_vitl | tree_t76_mvs_rig | prompt_sparse_noisy | 4770.7 | 4616.0 | 0.9109 | 0.056 | 106729.9 | 78.9291 | 0.0320 |
| promptda_vitl | tree_t76_mvs_rig | prompt_sparse_noisy_bgfill | 900.6 | 503.5 | 0.0997 | 0.894 | 870.2 | 0.9875 | 0.0319 |

Phase 1 real RGB-D sensor-noise reference floor:

```json
{
  "tree0_orbit16": {
    "n_valid_pixels": 166888,
    "median_scale_factor_pred_to_gt": 0.9968236349688758,
    "rmse_mm": 98.30747093850164,
    "mae_mm": 67.38997677486732,
    "median_abs_err_mm": 43.61271858215332,
    "abs_rel": 0.013120836822663748,
    "delta1": 1.0,
    "delta2": 1.0,
    "scaled_rmse_mm": 98.48802221311273,
    "scaled_mae_mm": 68.03582872303905,
    "scaled_median_abs_err_mm": 44.90270140152397,
    "scaled_abs_rel": 0.013184688097937899,
    "scaled_delta1": 1.0,
    "scaled_delta2": 1.0,
    "per_view_rmse_mm_mean": 98.02032657116813,
    "per_view_rmse_mm_min": 81.12333731409036,
    "per_view_rmse_mm_max": 123.17736629971051,
    "n_views_scored": 16
  },
  "tree_t76_mvs_rig": {
    "n_valid_pixels": 438773,
    "median_scale_factor_pred_to_gt": 0.9982827264630455,
    "rmse_mm": 93.92601273515231,
    "mae_mm": 62.55347476010652,
    "median_abs_err_mm": 39.17217254638672,
    "abs_rel": 0.012414790468160368,
    "delta1": 1.0,
    "delta2": 1.0,
    "scaled_rmse_mm": 93.72291810300507,
    "scaled_mae_mm": 62.54648919574594,
    "scaled_median_abs_err_mm": 39.36078434875245,
    "scaled_abs_rel": 0.012383763186756856,
    "scaled_delta1": 1.0,
    "scaled_delta2": 1.0,
    "per_view_rmse_mm_mean": 93.82915808003975,
    "per_view_rmse_mm_min": 74.71795365193825,
    "per_view_rmse_mm_max": 117.36748290331079,
    "n_views_scored": 42
  }
}
```
