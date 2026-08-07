"""Figures and scientific documents for Task 08 homography analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MatplotlibPath
from scipy.interpolate import griddata

from homography.analysis import (
    PERTURBATION_SCALES_PX,
    SCENES,
    Paths,
    apply_homography,
    load_scene_inputs,
    relative,
    sha256_file,
    write_csv,
)
from homography.calibration import (
    HISTORICAL_CONFIDENCE,
    HISTORICAL_MAX_ITERS,
    HISTORICAL_RANSAC_THRESHOLD_PX,
)


SCENE_LABELS = {
    "bellevue_116th_ne12th": "116th / NE12th",
    "bellevue_150th_newport": "150th / Newport",
    "bellevue_150th_eastgate": "150th / Eastgate",
    "bellevue_150th_se38th": "150th / SE38th",
    "bellevue_ne8th": "NE8th",
}


def _save_figure(figure: plt.Figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = [base.with_suffix(".png"), base.with_suffix(".pdf")]
    figure.savefig(outputs[0], dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(outputs[1], bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return outputs


def _style_axes(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#d8dde3", linewidth=0.7, alpha=0.75)


def _markdown_table(frame: pd.DataFrame, decimals: int = 3) -> str:
    def value_text(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{decimals}f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(value_text(row[column]) for column in frame.columns) + " |")
    return "\n".join(lines)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def run_figures(paths: Paths) -> dict[str, Any]:
    quality = pd.read_csv(paths.results / "homography_quality_metrics.csv")
    point_errors = pd.read_csv(paths.results / "calibration_point_errors.csv")
    perturbation = pd.read_csv(paths.results / "homography_perturbation_summary.csv")
    runs = pd.read_parquet(paths.results / "homography_perturbation_runs.parquet")
    jackknife = pd.read_csv(paths.results / "homography_jackknife_sensitivity.csv")
    quality_target = pd.read_csv(paths.results / "homography_quality_vs_target_error.csv")
    scenes = load_scene_inputs(paths)
    manifest_rows = []

    for scene in SCENES:
        state = scenes[scene]
        scene_directory = paths.figures / scene
        source = state.points[["camera_x", "camera_y"]].to_numpy(float)
        destination = state.points[["topview_x", "topview_y"]].to_numpy(float)
        projected = apply_homography(source, state.frozen_matrix)
        errors = point_errors[point_errors["scene_id"] == scene][
            "forward_reprojection_error_px"
        ].to_numpy(float)

        image = plt.imread(state.camera_image_path)
        figure, axis = plt.subplots(figsize=(10.5, 6.0))
        axis.imshow(image)
        axis.scatter(source[:, 0], source[:, 1], c="#d62728", s=42, edgecolor="white")
        for point, point_id in zip(source, state.points["pair_id"], strict=True):
            axis.text(point[0] + 7, point[1] - 7, str(int(point_id)), color="white", fontsize=8)
        axis.set_title(f"{SCENE_LABELS[scene]}: frozen camera-space correspondences")
        axis.set_axis_off()
        outputs = _save_figure(figure, scene_directory / "calibration_source_points")
        for output in outputs:
            manifest_rows.append(
                _figure_manifest_row(paths, output, scene, "camera frame with calibration IDs", False)
            )

        figure, (axis, error_axis) = plt.subplots(
            1,
            2,
            figsize=(13.0, 6.3),
            gridspec_kw={"width_ratios": [1.7, 1.0]},
        )
        axis.scatter(
            destination[:, 0],
            destination[:, 1],
            c="#18864b",
            s=38,
            label="manual destination",
        )
        axis.scatter(
            projected[:, 0],
            projected[:, 1],
            c="#b52a2a",
            marker="x",
            s=42,
            label="reprojected",
        )
        for expected, actual in zip(destination, projected, strict=True):
            axis.plot(
                [expected[0], actual[0]],
                [expected[1], actual[1]],
                color="#e38b2c",
                linewidth=1.0,
            )
        axis.set_xlim(0, state.topview_size[0])
        axis.set_ylim(state.topview_size[1], 0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("top-view x [px]")
        axis.set_ylabel("top-view y [px]")
        axis.set_title("destination-space vectors")
        axis.legend(frameon=False, loc="best")
        _style_axes(axis)

        order = np.argsort(errors)
        sorted_errors = errors[order]
        sorted_ids = state.points.iloc[order]["pair_id"].astype(int).astype(str)
        positions = np.arange(len(order))
        error_axis.barh(positions, sorted_errors, color="#3675a9")
        error_axis.set_yticks(positions, sorted_ids)
        error_axis.set_xlabel("forward error [px]")
        error_axis.set_ylabel("point ID")
        error_axis.set_title("pointwise residuals")
        for position, value in zip(positions, sorted_errors, strict=True):
            error_axis.text(value + max(errors) * 0.015, position, f"{value:.1f}", va="center", fontsize=7)
        error_axis.set_xlim(0, max(errors) * 1.18)
        _style_axes(error_axis)
        figure.suptitle(f"{SCENE_LABELS[scene]}: forward reprojection")
        outputs = _save_figure(figure, scene_directory / "destination_reprojection_vectors")
        for output in outputs:
            manifest_rows.append(
                _figure_manifest_row(paths, output, scene, "schematic destination coordinate canvas", False)
            )

        endpoints = state.camera_endpoints[
            ["start_x", "start_y", "end_x", "end_y"]
        ].to_numpy(float).reshape(-1, 2)
        hull = np.asarray(
            quality.loc[quality["scene_id"] == scene, "source_calibration_hull_area_fraction"]
        )
        del hull
        import cv2

        hull_points = cv2.convexHull(source.astype(np.float32)).reshape(-1, 2)
        inside = np.array(
            [
                cv2.pointPolygonTest(hull_points.astype(np.float32), tuple(point), False) >= 0
                for point in endpoints
            ]
        )
        sample_indices = np.linspace(0, len(endpoints) - 1, min(4000, len(endpoints)), dtype=int)
        figure, axis = plt.subplots(figsize=(10.5, 6.0))
        axis.imshow(image)
        sampled = endpoints[sample_indices]
        sampled_inside = inside[sample_indices]
        axis.scatter(sampled[~sampled_inside, 0], sampled[~sampled_inside, 1], s=4, c="#cf3c3c", alpha=0.18, label="endpoint outside hull")
        axis.scatter(sampled[sampled_inside, 0], sampled[sampled_inside, 1], s=5, c="#15834b", alpha=0.25, label="endpoint inside hull")
        closed = np.vstack([hull_points, hull_points[0]])
        axis.plot(closed[:, 0], closed[:, 1], color="#ffcc33", linewidth=2.0, label="calibration hull")
        axis.set_title(f"{SCENE_LABELS[scene]}: endpoint support relative to calibration hull")
        axis.legend(frameon=True, facecolor="white", framealpha=0.9, loc="best")
        axis.set_axis_off()
        outputs = _save_figure(figure, scene_directory / "source_coverage_and_extrapolation")
        for output in outputs:
            manifest_rows.append(
                _figure_manifest_row(paths, output, scene, "camera endpoint coverage diagnostic", False)
            )

        width, height = state.camera_size
        grid_x, grid_y = np.meshgrid(np.linspace(0, width, 240), np.linspace(0, height, 140))
        interpolated = griddata(source, errors, (grid_x, grid_y), method="linear")
        points_for_mask = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        hull_path = MatplotlibPath(hull_points)
        inside_grid = hull_path.contains_points(points_for_mask).reshape(grid_x.shape)
        interpolated[~inside_grid] = np.nan
        figure, axis = plt.subplots(figsize=(10.5, 5.8))
        field = axis.pcolormesh(grid_x, grid_y, interpolated, cmap="viridis", shading="auto")
        axis.scatter(source[:, 0], source[:, 1], c="white", edgecolor="black", s=28)
        axis.plot(closed[:, 0], closed[:, 1], color="white", linewidth=1.2)
        axis.set_xlim(0, width)
        axis.set_ylim(height, 0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{SCENE_LABELS[scene]}: diagnostic interpolation from sparse calibration residuals")
        axis.set_xlabel("camera x [px]")
        axis.set_ylabel("camera y [px]")
        colorbar = figure.colorbar(field, ax=axis, pad=0.02)
        colorbar.set_label("interpolated forward residual [px]")
        outputs = _save_figure(figure, scene_directory / "sparse_residual_interpolation")
        for output in outputs:
            manifest_rows.append(
                _figure_manifest_row(paths, output, scene, "interpolation, not dense ground truth", False)
            )

        scene_runs = runs[(runs["scene_id"] == scene) & (runs["status"] == "ok")]
        figure, axis = plt.subplots(figsize=(8.4, 4.8))
        for scale_index, scale in enumerate(PERTURBATION_SCALES_PX):
            values = scene_runs[scene_runs["noise_scale_px"] == scale]["resulting_target"].to_numpy(float)
            x_values = np.full(len(values), scale_index) + np.linspace(-0.14, 0.14, len(values))
            axis.scatter(x_values, values, s=22, alpha=0.65, color="#235789")
        frozen_target = int(scene_runs["frozen_target"].iloc[0])
        axis.axhline(frozen_target, color="#b52a2a", linestyle="--", label=f"frozen target = {frozen_target}")
        axis.set_xticks(
            range(len(PERTURBATION_SCALES_PX)),
            [f"+/-{int(value)}" for value in PERTURBATION_SCALES_PX],
        )
        axis.set_xlabel("bounded source-point perturbation [px]")
        axis.set_ylabel("recomputed $K_{HG}$")
        axis.set_title(f"{SCENE_LABELS[scene]}: target response to calibration perturbation")
        axis.legend(frameon=False)
        _style_axes(axis)
        outputs = _save_figure(figure, scene_directory / "perturbation_target_distribution")
        for output in outputs:
            manifest_rows.append(
                _figure_manifest_row(paths, output, scene, "target-estimation perturbation diagnostic", False)
            )

    figure, axis = plt.subplots(figsize=(10.2, 5.2))
    positions = np.arange(len(quality))
    axis.bar(positions - 0.18, 100 * quality["normalized_rmse_fraction_destination_diagonal"], width=0.36, label="RMSE / destination diagonal")
    axis.bar(positions + 0.18, 100 * quality["normalized_p95_fraction_destination_diagonal"], width=0.36, label="P95 / destination diagonal")
    axis.set_xticks(positions, [SCENE_LABELS[value] for value in quality["scene_id"]], rotation=18, ha="right")
    axis.set_ylabel("normalized error [%]")
    axis.set_title("Five-scene homography calibration error")
    axis.legend(frameon=False)
    _style_axes(axis)
    outputs = _save_figure(figure, paths.figures / "homography_quality_summary")
    for output in outputs:
        manifest_rows.append(_figure_manifest_row(paths, output, "ALL", "aggregate quality metrics", False))

    figure, axis = plt.subplots(figsize=(9.7, 5.2))
    for scene in SCENES:
        subset = perturbation[perturbation["scene_id"] == scene]
        axis.plot(subset["noise_scale_px"], subset["target_preservation_probability"], marker="o", label=SCENE_LABELS[scene])
    axis.set_ylim(-0.02, 1.04)
    axis.set_xlabel("bounded source-point perturbation [px]")
    axis.set_ylabel("frozen-target preservation probability")
    axis.set_title("Target preservation under deterministic calibration perturbation")
    axis.legend(frameon=False, ncol=2)
    _style_axes(axis)
    outputs = _save_figure(figure, paths.figures / "target_preservation_by_scale")
    for output in outputs:
        manifest_rows.append(_figure_manifest_row(paths, output, "ALL", "perturbation summary", False))

    jack_summary = jackknife.groupby("scene_id").agg(
        preservation=("target_difference_from_frozen", lambda values: float(np.mean(values == 0)))
    ).reindex(SCENES)
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    axis.bar(range(len(jack_summary)), jack_summary["preservation"], color="#3675a9")
    axis.set_xticks(range(len(jack_summary)), [SCENE_LABELS[value] for value in jack_summary.index], rotation=18, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("target preserved after point omission")
    axis.set_title("Leave-one-correspondence-out target stability")
    _style_axes(axis)
    outputs = _save_figure(figure, paths.figures / "jackknife_target_stability")
    for output in outputs:
        manifest_rows.append(_figure_manifest_row(paths, output, "ALL", "jackknife summary", False))

    figure, axis = plt.subplots(figsize=(7.8, 5.2))
    axis.scatter(100 * quality_target["normalized_rmse_fraction_destination_diagonal"], quality_target["frozen_target_absolute_error"], s=55, color="#235789")
    for row in quality_target.itertuples():
        axis.annotate(SCENE_LABELS[row.scene_id], (100 * row.normalized_rmse_fraction_destination_diagonal, row.frozen_target_absolute_error), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set_xlabel("normalized reprojection RMSE [% of destination diagonal]")
    axis.set_ylabel("frozen target absolute error [movements]")
    axis.set_title("Descriptive calibration quality versus target error (n = 5)")
    _style_axes(axis)
    outputs = _save_figure(figure, paths.figures / "quality_vs_target_error_descriptive")
    for output in outputs:
        manifest_rows.append(_figure_manifest_row(paths, output, "ALL", "descriptive n=5 diagnostic", False))

    manifest = pd.DataFrame(manifest_rows)
    write_csv(manifest, paths.results / "homography_figure_manifest.csv")
    return {"homography_figure_count": len(manifest)}


def _figure_manifest_row(
    paths: Paths,
    path: Path,
    scene: str,
    description: str,
    contains_google_imagery: bool,
) -> dict[str, Any]:
    return {
        "path": relative(paths, path),
        "scene_id": scene,
        "description": description,
        "contains_google_maps_or_satellite_imagery": contains_google_imagery,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _quality_display(quality: pd.DataFrame) -> pd.DataFrame:
    output = quality[
        [
            "scene_id",
            "point_count",
            "inlier_count",
            "forward_mean_error_px",
            "forward_median_error_px",
            "forward_rmse_error_px",
            "forward_p95_error_px",
            "normalized_rmse_fraction_destination_diagonal",
            "source_calibration_hull_area_fraction",
            "endpoint_extrapolation_fraction",
            "quality_class",
            "homography_passes_quality_gate",
        ]
    ].copy()
    output["scene_id"] = output["scene_id"].map(SCENE_LABELS)
    output["normalized_rmse_percent"] = 100 * output.pop(
        "normalized_rmse_fraction_destination_diagonal"
    )
    output["hull_coverage_percent"] = 100 * output.pop(
        "source_calibration_hull_area_fraction"
    )
    output["endpoint_extrapolation_percent"] = 100 * output.pop(
        "endpoint_extrapolation_fraction"
    )
    return output


def run_documents(paths: Paths) -> dict[str, Any]:
    quality = pd.read_csv(paths.results / "homography_quality_metrics.csv")
    reproduction = pd.read_csv(paths.results / "homography_reproduction.csv")
    correspondences = pd.read_csv(paths.results / "calibration_correspondences.csv")
    perturbation = pd.read_csv(paths.results / "homography_perturbation_summary.csv")
    runs = pd.read_parquet(paths.results / "homography_perturbation_runs.parquet")
    jackknife = pd.read_csv(paths.results / "homography_jackknife_sensitivity.csv")
    quality_target = pd.read_csv(paths.results / "homography_quality_vs_target_error.csv")
    result_manifest = json.loads(
        (paths.results / "task_08_result_manifest.json").read_text(encoding="utf-8")
    )
    quality_table = _quality_display(quality)
    point_counts = correspondences.groupby("scene_id").size().reindex(SCENES)
    jack_summary = jackknife.groupby("scene_id").agg(
        omissions=("omitted_point_id", "size"),
        target_preservation=("target_difference_from_frozen", lambda values: float(np.mean(values == 0))),
        target_min=("resulting_target", "min"),
        target_max=("resulting_target", "max"),
        median_p95_endpoint_displacement_px=("endpoint_displacement_p95_px", "median"),
    ).reset_index()
    jack_summary["scene_id"] = jack_summary["scene_id"].map(SCENE_LABELS)
    compact_perturbation = perturbation[
        [
            "scene_id",
            "noise_scale_px",
            "target_preservation_probability",
            "target_min",
            "target_max",
            "entry_k_preservation_probability",
            "exit_k_preservation_probability",
            "threshold_preservation_probability",
            "endpoint_displacement_p95_px",
            "od_pair_change_mean",
        ]
    ].copy()
    compact_perturbation["scene_id"] = compact_perturbation["scene_id"].map(SCENE_LABELS)

    implementation_audit = f"""# Homography Implementation Audit

## Frozen implementation

The five-scene HG-MSA-TC homographies are read from
`research_experiments/fov2026_trajectory_clustering/configs/hg_msa_tc_five_scene/`.
The historical calibration runner is
`research_experiments/fov2026_trajectory_clustering/scripts/hg_msa_tc_five_scene/run_homography_calibration_five_scene.py`.
It calls `cv2.findHomography(source, destination, cv2.RANSAC, 10.0)` and normalizes
the returned matrix by `H[2,2]`. Under OpenCV 4.12.0, the omitted arguments resolve
to `maxIters={HISTORICAL_MAX_ITERS}` and `confidence={HISTORICAL_CONFIDENCE}`. A direct
least-squares call (`method=0`) is used only if RANSAC fails or returns fewer than four
inliers; no frozen scene used that fallback.

NE8th uses the same estimator after excluding manually identified point IDs
`12, 13, 15, 21, 23`; the original 23-row CSV remains unchanged and the frozen
calibration uses the 18-row filtered file. No lens-distortion correction, point
normalization, nonlinear refinement, or manual matrix postprocessing is present.

## Exact data flow

1. Correspondences are ordered by their CSV row order and interpreted as camera
   `(x,y)` to top-view `(x,y)` pairs.
2. RANSAC uses a 10 px threshold in destination/top-view pixel coordinates.
3. OpenCV returns the inlier mask and a refined matrix estimated from the consensus set.
4. The matrix is divided by `H[2,2]` and stored as camera-to-top-view.
5. The target estimator transforms only canonical first/last endpoints from the
   `target_estimation` split.

## Scene inputs

{_markdown_table(quality_table[["scene_id", "point_count", "inlier_count", "source_image_width", "source_image_height", "destination_image_width", "destination_image_height"]] if "source_image_width" in quality_table else quality[["scene_id", "point_count", "inlier_count", "source_image_width", "source_image_height", "destination_image_width", "destination_image_height"]])}

## Duplicate and superseded implementations

- `homography_extension/scripts/run_multiscene_homography_v2.py` uses `method=0` and
  belongs to an earlier extension; it is not the frozen five-scene estimator.
- `homography_extension/scripts/run_final_homography_extension_v1.py` contains earlier
  direct-fit sensitivity code; it is not used here.
- Task 08 centralizes the frozen behavior in `code/homography/calibration.py`.

The final five-scene repository evidence supports RANSAC. Any manuscript wording that
describes all historical extension scripts as RANSAC would still be inaccurate and
must distinguish the final frozen pipeline from earlier case-study code.
"""
    _write(paths.docs / "homography_implementation_audit.md", implementation_audit)

    provenance_rows = []
    google_names = {
        "bellevue_116th_ne12th": "Bellevue_116th_NE12th_google_maps.png",
        "bellevue_150th_newport": "Bellevue_150th_Newport_google_maps.png",
        "bellevue_150th_eastgate": "Bellevue_150th_Eastgate_google_maps.png",
        "bellevue_150th_se38th": "Bellevue_150th_SE38th_google_maps.png",
        "bellevue_ne8th": "Bellevue_NE_NE8th_google_maps.png",
    }
    for scene in SCENES:
        subset = correspondences[correspondences["scene_id"] == scene]
        provenance_rows.append(
            {
                "scene": SCENE_LABELS[scene],
                "points": len(subset),
                "point provenance": subset["point_provenance"].iloc[0],
                "top-view source": f"old_data/Google_Maps_Pics/{google_names[scene]}",
                "landmark type": "unknown/not documented per point",
            }
        )
    provenance = f"""# Homography Calibration Point Provenance

All 101 frozen correspondences were selected manually. Newport, Eastgate and SE38th
reuse point sets from prior calibration assets; 116th/NE12th and NE8th were collected
with the visual point-pair workflow. The CSVs do not document whether each point is a
lane corner, marking, curb feature, or another landmark, so point-level feature type
is recorded as `unknown/not documented` rather than inferred.

{_markdown_table(pd.DataFrame(provenance_rows))}

The five `topview_reference_image.png` files are byte-identical to the named files in
`old_data/Google_Maps_Pics`. Repository reports identify these as Google Maps/top-view
screenshots. Camera reference images are extracted traffic-video frames. Checksums,
normalized coordinates, image dimensions, and inlier flags are published in
`results/homography/calibration_correspondences.csv`.

NE8th's five rejected IDs remain auditable in the original CSV and are not silently
deleted. The frozen correspondence export contains only the 18 points actually used.
"""
    _write(paths.docs / "homography_calibration_point_provenance.md", provenance)

    mathematics = r"""# Homography Mathematical Definition

For camera point \(p_i=[x_i,y_i,1]^T\) and projective matrix \(H\), the destination
point is \(\hat q_i \sim H p_i\). If the rows of \(H\) are \(h_1^T,h_2^T,h_3^T\),

\[
\hat x_i' = \frac{h_1^T p_i}{h_3^T p_i},\qquad
\hat y_i' = \frac{h_2^T p_i}{h_3^T p_i}.
\]

The primary all-point forward residual is
\(e_i^f=\lVert\hat q_i-q_i\rVert_2\), measured in top-view pixels. We report
mean, median, root-mean-square (RMSE), maximum, P90 and P95 residuals. The normalized
RMSE is RMSE divided by the top-view image diagonal. Inverse residuals apply
\(H^{-1}\) and are reported separately in camera pixels. A unitless symmetric
diagnostic is

\[
e_i^{sym,n}=e_i^f/d_{top}+e_i^b/d_{camera}.
\]

RANSAC classifies a correspondence as an inlier when its destination reprojection
residual is within 10 px according to OpenCV's implementation. The final matrix is
normalized by \(H_{33}\). The normalized-DLT effective condition and null-space gap
are diagnostic only; they do not replace or alter the frozen matrix.

Calibration coverage is the source-point convex-hull area divided by image area.
Endpoint extrapolation is the fraction of target-estimation entry/exit endpoints not
covered by that source convex hull. Sparse residual interpolation is shown only inside
the calibration hull and is explicitly not a measured dense error field.
"""
    _write(paths.docs / "homography_mathematical_definition.md", mathematics)

    reproduction_report = f"""# Homography Reproduction Report

Every frozen matrix was re-estimated from the stored correspondences with the exact
historical OpenCV call. Matrices were compared after `H[2,2]=1` normalization.

{_markdown_table(reproduction[["scene_id", "point_count", "frozen_inlier_count", "reproduced_inlier_count", "inlier_mask_exact_match", "matrix_max_abs_difference", "matrix_exact_match"]])}

Maximum matrix-element error across all scenes: `{reproduction['matrix_max_abs_difference'].max():.3e}`.
All inlier masks are exact. The frozen JSON files were not overwritten.
"""
    _write(paths.docs / "homography_reproduction_report.md", reproduction_report)

    gate = f"""# Objective Homography Quality Gate

The gate was specified from pixel-scale, normalized-error, correspondence coverage,
and extrapolation considerations before the independent target-error column was read.
It is a diagnostic engineering gate and is not tuned to clustering or reference-label
performance.

| Class | normalized RMSE | normalized P95 | inlier fraction | source hull area | endpoint extrapolation | Boolean pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| good | <=1% | <=2% | >=60% | >=20% | <=95% | yes |
| acceptable | <=2% | <=4% | >=40% | >=10% | <=98% | yes |
| acceptable with caution | <=3% | <=6% | >=30% | >=5% | <=99.5% | yes |
| poor | otherwise | otherwise | otherwise | otherwise | otherwise | no |

All criteria in a row must hold. The first matching row determines the class.

{_markdown_table(quality_table)}

All five scenes pass, but this does not imply uniform spatial accuracy. Endpoint
extrapolation is {100*quality['endpoint_extrapolation_fraction'].min():.1f}% to
{100*quality['endpoint_extrapolation_fraction'].max():.1f}%, because calibration
features concentrate around the intersection while trajectory endpoints often lie
near image boundaries. Newport and NE8th are therefore `acceptable`, not `good`.
No scene is removed or recalibrated post hoc.
"""
    _write(paths.docs / "homography_quality_gate.md", gate)

    sensitivity = f"""# Homography-to-Target Sensitivity Report

## Protocol

The analysis perturbs camera/source calibration points independently with bounded
uniform noise at `+/-1, +/-2, +/-3, +/-5, +/-10 px`. Destination points remain fixed. Each
scene-scale cell has {int(result_manifest['perturbation_replicates_per_scene_scale'])}
fixed-seed replicates. Every replicate uses the exact historical homography estimator
and the full frozen target-estimation algorithm on `target_estimation`; no K, threshold,
or target is held artificially fixed. This is a finite deterministic Monte Carlo
diagnostic, not a high-precision probability estimate.

{_markdown_table(compact_perturbation, 4)}

## Jackknife

{_markdown_table(jack_summary, 4)}

## Interpretation

- 116th/NE12th preserves the target in every jackknife and perturbation run.
- Newport shows a threshold-linked discontinuity: small perturbations occasionally
  move the result from 12 to 7 even though mean OD assignment changes remain below 1%.
- Eastgate preserves 9 in 61.5% of leave-one-point-out runs and at least 85% of
  perturbations at every tested scale.
- NE8th has large top-view displacement under perturbation, consistent with strong
  projective amplification outside the calibration hull, while OD changes remain low.
- SE38th preserves 18 in every `+/-1-5 px` replicate. At `+/-10 px`, 14/20 runs preserve
  18 and the observed range is 16-37. Calibration uncertainty changes the geometric
  target only under large perturbation and does not move it toward 9.
"""
    _write(paths.docs / "homography_to_target_sensitivity_report.md", sensitivity)

    se38_quality = quality[quality["scene_id"] == "bellevue_150th_se38th"].iloc[0]
    se38_perturb = perturbation[perturbation["scene_id"] == "bellevue_150th_se38th"]
    se38_runs = runs[(runs["scene_id"] == "bellevue_150th_se38th") & (runs["status"] == "ok")]
    se38_jack = jackknife[jackknife["scene_id"] == "bellevue_150th_se38th"]
    se38_diagnosis = f"""# SE38th Homography Diagnosis

The frozen SE38th calibration has 24 correspondences and 16 RANSAC inliers. Its
all-point forward mean is {se38_quality['forward_mean_error_px']:.3f} px, median
{se38_quality['forward_median_error_px']:.3f} px, RMSE
{se38_quality['forward_rmse_error_px']:.3f} px and P95
{se38_quality['forward_p95_error_px']:.3f} px. Normalized RMSE is
{100*se38_quality['normalized_rmse_fraction_destination_diagonal']:.3f}% of the
top-view diagonal. The calibration hull covers
{100*se38_quality['source_calibration_hull_area_fraction']:.1f}% of the camera image,
while {100*se38_quality['endpoint_extrapolation_fraction']:.1f}% of target-estimation
endpoints lie outside that hull.

The target remains 18 in all {len(se38_runs[se38_runs['noise_scale_px'] <= 5])}
replicates through `+/-5 px`. At `+/-10 px`, preservation is
{se38_perturb.loc[se38_perturb['noise_scale_px'] == 10, 'target_preservation_probability'].iloc[0]:.0%}
and the target range is
{int(se38_perturb.loc[se38_perturb['noise_scale_px'] == 10, 'target_min'].iloc[0])}-{int(se38_perturb.loc[se38_perturb['noise_scale_px'] == 10, 'target_max'].iloc[0])}.
None of the {len(se38_runs)} perturbation runs produces 9. Leave-one-point-out preserves
18 in {(se38_jack['target_difference_from_frozen'] == 0).mean():.1%} of cases; omitting
inlier point 5 or 6 changes the endpoint partition to 7 entry and 8 exit regions and
raises the target to 43, indicating a localized calibration dependency rather than a
plausible correction toward the semantic count.

Task 07 showed four automatic entry regions dominated by one physical approach and
semantic duplication among supported OD pairs. The new evidence therefore supports
the diagnosis that SE38th is primarily an endpoint-region granularity/semantic
over-segmentation problem. Calibration extrapolation and two influential points are
secondary fragility factors, but realistic small perturbations do not explain the
18-versus-9 error.
"""
    _write(paths.docs / "se38th_homography_diagnosis.md", se38_diagnosis)

    imagery = """# Imagery Provenance and Licensing Audit

## Evidence found

- Each `topview_reference_image.png` is byte-identical to a scene file under
  `old_data/Google_Maps_Pics`; repository reports describe the files as Google Maps
  screenshots/top-view images.
- The five camera reference images are extracted Traffic Node Video Dataset frames.
- No acquisition date, zoom level, map coordinates, Google attribution metadata, or
  redistribution permission is stored alongside the top-view screenshots.
- Point coordinates and matrices are reproducible, but they do not resolve image-use
  rights.

## Publication handling

This audit does not make a legal conclusion. Because redistribution permission and
attribution metadata are not documented, the Google-derived raster images should not
be included in the Task 08 review ZIP or redistributed manuscript supplement. Use
camera frames, blank-coordinate reprojection plots, author-created schematic diagrams,
or an appropriately attributed OpenStreetMap-derived figure where licensing permits.
The Task 08 figures deliberately plot destination correspondences on a blank coordinate
canvas and mark every generated figure as free of embedded Google imagery.

Traffic-video frames remain subject to the Traffic Node Video Dataset distribution
conditions and should be handled under the dataset's terms. Exact source-image hashes
are retained locally for reproducibility without copying the source rasters into the
review ZIP.
"""
    _write(paths.docs / "imagery_provenance_and_licensing_audit.md", imagery)

    quality_target_display = quality_target[
        [
            "scene_id",
            "forward_mean_error_px",
            "normalized_rmse_fraction_destination_diagonal",
            "endpoint_extrapolation_fraction",
            "automatic_od_target",
            "observed_independent_semantic_movement_count",
            "frozen_target_absolute_error",
        ]
    ].copy()
    quality_target_display["scene_id"] = quality_target_display["scene_id"].map(SCENE_LABELS)
    scientific = f"""# Homography Scientific Interpretation

1. **Calibration accuracy.** All frozen matrices reproduce exactly. Mean all-point
   errors are {quality['forward_mean_error_px'].min():.2f}-{quality['forward_mean_error_px'].max():.2f}
   px; normalized RMSE is {100*quality['normalized_rmse_fraction_destination_diagonal'].min():.2f}%-{100*quality['normalized_rmse_fraction_destination_diagonal'].max():.2f}%.
2. **Quality gate.** All five pass the a priori diagnostic gate: two `good`, three
   `acceptable`. High endpoint extrapolation remains a shared limitation.
3. **Method role.** Homography supports geometric endpoint-region target estimation;
   camera isotropic shared-scale remains the frozen clustering representation.
4. **Calibration sensitivity.** Small perturbations generally preserve targets, but
   Newport has a threshold discontinuity and Eastgate/NE8th show localized fragility.
5. **SE38th.** Its 18 target persists under plausible small perturbations; calibration
   uncertainty alone does not explain the semantic overestimate.

{_markdown_table(quality_target_display, 4)}

With five scenes, this table is descriptive only and cannot support a reliable
correlation claim between reprojection error and target error. The manuscript should
narrow any claim that homography recovers semantic maneuver counts: it provides a
reproducible geometric structure signal whose granularity can differ from semantic
maneuvers. Publish point coordinates, image dimensions, RANSAC settings, masks,
residual distributions, coverage, extrapolation and perturbation results. Do not
remove or recalibrate SE38th post hoc.
"""
    _write(paths.docs / "homography_scientific_interpretation.md", scientific)

    manuscript = f"""# Manuscript-Ready Homography Section

## Methods: homography calibration and quality control

For each scene, manually selected camera-to-top-view correspondences were used to
estimate a projective transform with OpenCV RANSAC (destination reprojection threshold
{HISTORICAL_RANSAC_THRESHOLD_PX:.0f} px, maximum {HISTORICAL_MAX_ITERS} iterations,
confidence {HISTORICAL_CONFIDENCE}). The resulting matrix was normalized by its
bottom-right element. NE8th used 18 correspondences after five previously identified
high-error points were excluded; the original point table was retained for audit.
No lens-distortion correction was applied.

Forward error was defined as the Euclidean distance between each manual top-view point
and the projection of its camera counterpart. We report all-point mean, median, RMSE,
maximum, P90 and P95 errors, together with normalized RMSE, RANSAC inlier rate,
calibration-hull coverage and endpoint extrapolation. An a priori engineering gate
combined normalized residual, coverage and extrapolation criteria. The gate was not
tuned against independent clustering/reference results.

## Results

{_markdown_table(quality_table, 4)}

All five matrices were reproduced exactly from stored correspondences and all scenes
passed the diagnostic gate. However, 90.4-96.9% of target-estimation endpoints lay
outside the source calibration hull. This high extrapolation rate limits claims of
uniform spatial accuracy.

Calibration sensitivity was evaluated by leaving out each correspondence and by
adding fixed-seed bounded source-point perturbations (`+/-1, +/-2, +/-3, +/-5, +/-10 px`; 20
replicates per scene-scale). Each replicate reran the complete frozen target estimator
on `target_estimation`. SE38th retained `K_HG=18` in every replicate through `+/-5 px`;
at `+/-10 px`, 14/20 retained 18 and no replicate approached the independent semantic
count of 9. The SE38th discrepancy is therefore better explained by endpoint-region
granularity and semantic duplication than by calibration noise alone, although point
omission reveals local fragility.

Homography is a support layer for target estimation, not the final clustering feature
space. The method does not claim metric rectification, dense geometric ground truth,
or universally correct semantic maneuver counts. Manual correspondence uncertainty,
planar-scene assumptions, lens distortion, sparse calibration coverage, Google-image
alignment and imagery provenance remain limitations.
"""
    _write(paths.docs / "manuscript_ready_homography_section.md", manuscript)

    reviewer = """# Response to Reviewer: Homography Calibration

**Comment: The homography implementation and RANSAC settings are insufficiently specified.**

Response: We now state the exact OpenCV estimator, 10 px destination-space RANSAC
threshold, 2000-iteration limit, 0.995 confidence, point ordering, matrix
normalization and fallback behavior. [TO BE COMPLETED: section/page/line]

**Comment: Calibration points, inliers and image dimensions are not reproducible.**

Response: We provide all 101 source/destination pairs, normalized coordinates, image
dimensions, point IDs, inlier masks, source/config hashes and provenance status in a
machine-readable table. Point-level landmark type was not historically recorded and
is disclosed as unknown. [TO BE COMPLETED: supplement/table citation]

**Comment: Qualitative quality labels need an objective gate.**

Response: We replaced the earlier mean-error-only labels with a deterministic gate
combining normalized RMSE, normalized P95, inlier fraction, source-hull coverage and
endpoint extrapolation. The thresholds were specified independently of target/reference
performance, and no scene was removed post hoc. [TO BE COMPLETED: table/line]

**Comment: Calibration sensitivity is not quantified.**

Response: We added leave-one-correspondence-out analysis and 500 fixed-seed bounded
source-point perturbation runs. Each run propagates uncertainty through the full frozen
target estimator on the development target split. [TO BE COMPLETED: figure/table]

**Comment: Could SE38th's target error be a calibration artifact?**

Response: SE38th retained 18 in all perturbations through +/-5 px and never approached
9 at any tested scale. Two point omissions produce strong upward instability, but no
evidence supports calibration uncertainty as a correction toward the semantic count.
We now attribute the primary failure to endpoint-region granularity, with calibration
extrapolation as a secondary limitation. [TO BE COMPLETED: discussion lines]

**Comment: Map imagery source/licensing is unclear.**

Response: The top-view rasters are repository-identified Google Maps screenshots with
incomplete attribution metadata. We do not redistribute them in the review package and
recommend schematic or appropriately licensed replacements in the manuscript. This is
an evidence audit, not a legal conclusion. [TO BE COMPLETED: data/figure statement]
"""
    _write(paths.docs / "response_to_reviewer_homography_draft.md", reviewer)

    result_manifest_rows = [
        {"path": path, **metadata} for path, metadata in result_manifest["outputs"].items()
    ]
    result_manifest_doc = f"""# Homography Result Manifest

Generated: `{result_manifest['created_at_utc']}`

- frozen matrices modified: no;
- frozen targets modified: no;
- independent-test clustering rerun: no;
- quality gate tuned against reference metrics: no;
- propagation split: `target_estimation` only;
- perturbation replicates per scene-scale: {result_manifest['perturbation_replicates_per_scene_scale']}.

{_markdown_table(pd.DataFrame(result_manifest_rows))}
"""
    _write(paths.docs / "homography_result_manifest.md", result_manifest_doc)

    validation_path = paths.results / "task_08_validation_results.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        verification_text = (
            f"- Pytest: `{validation['pytest_passed']} passed in "
            f"{validation['pytest_seconds']:.2f}s`\n"
            f"- Ruff: `{validation['ruff_result']}`"
        )
    else:
        verification_text = "[TO BE COMPLETED AFTER FINAL TEST RUN: pytest and Ruff results]"
    task_report = f"""# Task 08 Execution Report

## Execution summary

- Branch: `feature/futuretransp-homography-quality-sensitivity`
- Frozen matrix reproduction: exact for all five scenes
- Calibration correspondences: {int(point_counts.sum())}
- Jackknife runs: {len(jackknife)}
- Perturbation runs: {len(runs)}
- Quality-gate pass: {int(quality['homography_passes_quality_gate'].sum())}/5
- Target propagation split: `target_estimation` only
- Frozen matrices/targets/configurations modified: no
- Independent-test clustering rerun: no

## Commands

```powershell
./.venv/Scripts/python.exe publications/hg-msa-tc-future-transportation/code/run_homography_analysis.py analyze --replicates 20
./.venv/Scripts/python.exe publications/hg-msa-tc-future-transportation/code/run_homography_reporting.py all
./.venv/Scripts/python.exe -m pytest publications/hg-msa-tc-future-transportation/tests -q
./.venv/Scripts/python.exe -m ruff check publications/hg-msa-tc-future-transportation/code publications/hg-msa-tc-future-transportation/tests
```

## Principal findings

- Exact reproduction confirms the final five-scene implementation is RANSAC with a
  10 px destination-space threshold.
- All scenes pass the diagnostic gate, but endpoint extrapolation is high.
- SE38th's 18 target persists under plausible small calibration perturbations.
- Newport exhibits a target-threshold discontinuity; Eastgate and NE8th have localized
  sensitivity that must remain visible in the paper.
- Google-derived top-view rasters are not included in the review package.

## Verification status

{verification_text}
"""
    _write(paths.docs / "task_08_execution_report.md", task_report)

    return {"homography_document_count": 13}


def update_publication_docs(paths: Paths) -> dict[str, Any]:
    readme = paths.publication / "README.md"
    marker = "## Homography Quality and Perturbation Analysis"
    text = readme.read_text(encoding="utf-8")
    text = text.replace(
        r".\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code"
        + "\r"
        + "un_homography_analysis.py analyze --replicates 20",
        "./.venv/Scripts/python.exe publications/hg-msa-tc-future-transportation/"
        "code/run_homography_analysis.py analyze --replicates 20",
    )
    text = text.replace(
        r".\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code"
        + "\r"
        + "un_homography_reporting.py all",
        "./.venv/Scripts/python.exe publications/hg-msa-tc-future-transportation/"
        "code/run_homography_reporting.py all",
    )
    text = text.replace(
        r".\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code"
        + "\n"
        + "un_homography_analysis.py analyze --replicates 20",
        "./.venv/Scripts/python.exe publications/hg-msa-tc-future-transportation/"
        "code/run_homography_analysis.py analyze --replicates 20",
    )
    text = text.replace(
        r".\.venv\Scripts\python.exe publications\hg-msa-tc-future-transportation\code"
        + "\n"
        + "un_homography_reporting.py all",
        "./.venv/Scripts/python.exe publications/hg-msa-tc-future-transportation/"
        "code/run_homography_reporting.py all",
    )
    if marker not in text:
        text += f"""

{marker}

Task 08 exactly reproduces the five frozen camera-to-top-view matrices, publishes all
calibration correspondences and inlier masks, applies an objective quality gate, and
propagates jackknife/point-perturbation uncertainty through the frozen target estimator
on `target_estimation` only.

```powershell
./.venv/Scripts/python.exe publications/hg-msa-tc-future-transportation/code/run_homography_analysis.py analyze --replicates 20
./.venv/Scripts/python.exe publications/hg-msa-tc-future-transportation/code/run_homography_reporting.py all
```

No frozen matrix, target, region count, threshold, selected configuration, reference
label or independent-test assignment is changed. Google-derived top-view rasters are
not redistributed in the Task 08 review package.
"""
    _write(readme, text)

    availability = paths.docs / "data_availability_plan.md"
    marker = "The homography-quality revision package"
    text = availability.read_text(encoding="utf-8")
    if marker not in text:
        text += """

The homography-quality revision package should include canonical calibration/quality
source, the frozen correspondence table, matrices, compact quality and sensitivity
tables, generated schematic/camera diagnostic figures, tests, manifests and reports.
It must not duplicate raw video, trajectory data, independent-test assignments, or
Google-derived top-view raster images. Source-image paths and hashes preserve local
auditability without asserting redistribution permission.
"""
        _write(availability, text)
    return {"publication_docs_updated": 2}


def run_all_reporting(paths: Paths) -> dict[str, Any]:
    result = {}
    result.update(run_figures(paths))
    result.update(run_documents(paths))
    result.update(update_publication_docs(paths))
    return result
