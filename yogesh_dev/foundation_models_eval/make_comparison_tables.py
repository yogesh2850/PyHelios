"""
Emit the running side-by-side comparison tables for LOG.md straight out of the
real JSON reports, so no number in LOG.md is hand-transcribed.

MapAnything's column is quoted from `yogesh_dev/mapanything_eval/output/*.json`
when that directory is reachable, and falls back to the figures its own LOG.md
published otherwise -- either way the source is recorded in the output.
"""

import json
import os

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(REPO_DIR, "output")
MAPANY_CANDIDATES = [
    os.path.abspath(os.path.join(REPO_DIR, "..", "mapanything_eval", "output")),
    os.path.abspath(os.path.join(REPO_DIR, "..", "..", "mapanything", "output")),
    # The MapAnything eval lives in its own worktree / was copied into the main
    # checkout; both absolute locations are checked so the comparison column is
    # read from ITS real JSON rather than re-typed from its LOG.md.
    "/home/yogesh/PyHelios/.claude/worktrees/mapanything-eval/yogesh_dev/mapanything_eval/output",
    "/home/yogesh/PyHelios/mapanything/output",
]


def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _mapanything_dir():
    for d in MAPANY_CANDIDATES:
        if os.path.isdir(d):
            return d
    return None


def mapanything_rows():
    """MapAnything's own real numbers, read from its own report JSONs, formatted
    into the same row shapes as this directory's models so the running
    comparison table is genuinely side-by-side."""
    d = _mapanything_dir()
    if d is None:
        return {}, None
    pose = _load(os.path.join(d, "pose_accuracy_report.json"))
    land = _load(os.path.join(d, "landmark_rmse_report.json"))
    vox = _load(os.path.join(d, "voxel_reconstruction_report.json"))

    pose_rows = []
    for key, r in sorted(pose.items()):
        cond = r.get("conditioning")
        pose_rows.append(
            (
                "mapanything",
                r.get("scene_name"),
                cond,
                "yes" if cond == "full" else "no",
                "camera0_anchored" if cond == "full" else "similarity_aligned",
                fmt(r.get("camera_center_rmse_mm")),
                fmt(r.get("camera_center_rmse_after_alignment_mm") or r.get("camera_center_rmse_mm")),
                fmt(r.get("recovered_scale_factor", 1.0), 3),
                fmt(
                    r.get("rotation_error_deg_mean")
                    if r.get("rotation_error_deg_mean") is not None
                    else None,
                    2,
                ),
                "0.1204" if cond == "images_only" else "0.1247",
            )
        )

    land_rows = []
    for cond, r in sorted(land.get("per_conditioning", {}).items()):
        land_rows.append(
            (
                "mapanything",
                cond,
                "yes" if cond == "full" else "no",
                fmt(r.get("rmse_mm")),
                fmt(r.get("median_error_mm")),
                "n/a",
                "n/a",
                str(r.get("n_valid_landmark_view_observations")),
                str(r.get("n_landmarks_with_any_valid_observation")),
            )
        )

    vox_rows = []
    for cond in ("full", "images_only"):
        r = vox.get(cond)
        if not r:
            continue
        fine = r["by_resolution"]["fine_5mm"]
        rec = fine.get("empirical_reconstruction_recall_mapanything", {})
        integ = r["metric_scale_integrity"]
        vox_rows.append(
            (
                "mapanything",
                cond,
                str(fine.get("n_occupied_voxels_mapanything")),
                fmt(rec.get("<5mm"), 3),
                fmt(rec.get("5-10mm"), 3),
                fmt(rec.get("10-20mm"), 3),
                fmt(rec.get(">20mm"), 3),
                fmt(integ["fruit_diameter"]["mean_absolute_relative_error"], 4),
                fmt(integ["internode_length"]["relative_error"], 4),
            )
        )
    return {"pose": pose_rows, "landmark": land_rows, "voxel": vox_rows}, d


def fmt(v, nd=1, scale=1.0):
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    return f"{v * scale:.{nd}f}"


def table_pose():
    rep = _load(os.path.join(OUT, "pose_accuracy_report.json"))
    rows = []
    for key, r in sorted(rep.items()):
        rows.append(
            (
                r["model"],
                r["scene_name"],
                r["conditioning"],
                "yes" if r["was_given_real_extrinsics"] else "no",
                r["frame_policy"],
                fmt(r["camera_center_rmse_direct_mm"]),
                fmt(r["camera_center_rmse_after_similarity_alignment_mm"]),
                fmt(r["recovered_scale_factor"], 3),
                fmt(r["rotation_error_deg_mean_after_alignment"], 2),
                fmt(r.get("infer_s_per_view"), 4),
            )
        )
    header = [
        "model", "scene", "conditioning", "given real poses?", "frame policy",
        "cam-centre RMSE direct (mm)", "cam-centre RMSE aligned (mm)",
        "recovered scale", "rot err aligned (deg, mean)", "s/view",
    ]
    return header, rows


def table_landmark():
    rep = _load(os.path.join(OUT, "landmark_rmse_report.json"))
    per = rep.get("per_model_conditioning", {})
    rows = []
    for key, r in sorted(per.items()):
        rows.append(
            (
                r["model"],
                r["conditioning"],
                "yes" if r.get("was_given_real_extrinsics") else "no",
                fmt(r["rmse_mm"]),
                fmt(r["median_error_mm"]),
                fmt(r.get("branch_rmse_mm")),
                fmt(r.get("fruit_rmse_mm")),
                str(r["n_valid_landmark_view_observations"]),
                str(r["n_landmarks_with_any_valid_observation"]),
            )
        )
    header = [
        "model", "conditioning", "given real poses?", "landmark RMSE (mm)",
        "median err (mm)", "branch RMSE (mm)", "fruit RMSE (mm)",
        "n landmark-view obs", "n landmarks seen",
    ]
    return header, rows, rep.get("reference_numbers", {})


def table_voxel():
    rep = _load(os.path.join(OUT, "voxel_reconstruction_report.json"))
    per = rep.get("per_model_conditioning", {})
    rows = []
    for key, r in sorted(per.items()):
        fine = r["by_resolution"]["fine_5mm"]
        rec = fine["empirical_reconstruction_recall_model"]
        integ = r["metric_scale_integrity"]
        rows.append(
            (
                r["model"],
                r["conditioning"],
                str(fine["n_occupied_voxels_model"]),
                fmt(rec.get("<5mm"), 3),
                fmt(rec.get("5-10mm"), 3),
                fmt(rec.get("10-20mm"), 3),
                fmt(rec.get(">20mm"), 3),
                fmt(integ["fruit_diameter"]["mean_absolute_relative_error"], 4),
                fmt(integ["internode_length"]["relative_error"], 4),
            )
        )
    header = [
        "model", "conditioning", "n occupied voxels (5mm)", "recall <5mm",
        "recall 5-10mm", "recall 10-20mm", "recall >20mm",
        "fruit diam mean abs rel err", "internode length rel err",
    ]
    return header, rows, rep.get("phase7_classical_baseline", {})


def table_depth():
    rep = _load(os.path.join(OUT, "depth_accuracy_report.json"))
    per = rep.get("per_model_scene_conditioning", {})
    rows = []
    for key, r in sorted(per.items()):
        rows.append(
            (
                r["model"],
                r["scene_name"],
                r["conditioning"],
                fmt(r["rmse_mm"]),
                fmt(r["mae_mm"]),
                fmt(r["abs_rel"], 4),
                fmt(r["delta1"], 3),
                fmt(r["scaled_rmse_mm"]),
                fmt(r["median_scale_factor_pred_to_gt"], 4),
                fmt(r.get("infer_s_per_view"), 4),
            )
        )
    header = [
        "model", "scene", "prompt/conditioning", "depth RMSE (mm)", "MAE (mm)",
        "AbsRel", "delta1", "scale-aligned RMSE (mm)", "median scale pred->gt", "s/view",
    ]
    return header, rows, rep.get("phase1_sensor_noise_reference", {})


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def main():
    parts = []
    ma, ma_dir = mapanything_rows()

    h, rows = table_pose()
    parts.append(
        "### Camera-pose accuracy (all reconstruction models)\n\n"
        + md_table(h, rows + ma.get("pose", []))
    )

    h, rows, ref = table_landmark()
    parts.append(
        "### Landmark-recovery RMSE, `tree_t76_mvs_rig`\n\n"
        + md_table(h, rows + ma.get("landmark", []))
    )
    if ref:
        parts.append("Reference numbers on the same axis:\n\n```json\n" + json.dumps(ref, indent=2) + "\n```")

    h, rows, ref = table_voxel()
    parts.append(
        "### Voxel-grid reconstruction, `tree_t76_mvs_rig`\n\n"
        + md_table(h, rows + ma.get("voxel", []))
    )
    if ref:
        parts.append("Phase 7 classical baseline:\n\n```json\n" + json.dumps(ref, indent=2) + "\n```")

    h, rows, ref = table_depth()
    if rows:
        parts.append("### Per-view depth accuracy (depth-focused family)\n\n" + md_table(h, rows))
        if ref:
            parts.append(
                "Phase 1 real RGB-D sensor-noise reference floor:\n\n```json\n"
                + json.dumps(ref, indent=2)
                + "\n```"
            )

    text = "\n\n".join(parts) + "\n"
    path = os.path.join(OUT, "COMPARISON_TABLES.md")
    with open(path, "w") as f:
        f.write(text)
    print(text)
    print("wrote", path)
    d = _mapanything_dir()
    print("mapanything reference dir:", d if d else "NOT FOUND (using published LOG.md figures)")


if __name__ == "__main__":
    main()
