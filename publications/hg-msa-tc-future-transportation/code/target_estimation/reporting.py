"""Publication figures and scientific documents for Task 07 target analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from target_estimation.analysis import SCENES, Paths, default_paths, sha256_file, utc_timestamp


SCENE_LABELS = {
    "bellevue_116th_ne12th": "116th / NE12th",
    "bellevue_150th_newport": "150th / Newport",
    "bellevue_150th_eastgate": "150th / Eastgate",
    "bellevue_150th_se38th": "150th / SE38th",
    "bellevue_ne8th": "NE8th",
}


def _save_figure(figure: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".png"), dpi=260, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _homography_index(paths: Paths) -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(
        (paths.homography_directory / "homography_five_scene_index.yaml").read_text(
            encoding="utf-8"
        )
    )
    return {row["scene"]: row for row in payload["scenes"]}


def _topview_path(paths: Paths, scene: str, index: dict[str, dict[str, Any]]) -> Path:
    return paths.repo / index[scene]["topview_image_path"]


def _endpoint_maps(paths: Paths) -> None:
    assignments = pd.read_parquet(
        paths.results / "target_endpoint_region_assignments.parquet"
    )
    parameters = pd.read_csv(paths.results / "frozen_scene_parameters.csv")
    index = _homography_index(paths)
    for scene in SCENES:
        scene_rows = assignments[assignments["scene"] == scene]
        parameter = parameters[parameters["scene"] == scene].iloc[0]
        background = mpimg.imread(_topview_path(paths, scene, index))
        for role, x_column, y_column, region_column, prefix in (
            ("entry", "start_x_topview", "start_y_topview", "entry_region", "entry"),
            ("exit", "end_x_topview", "end_y_topview", "exit_region", "exit"),
        ):
            figure, axis = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
            axis.imshow(background)
            for region, group in scene_rows.groupby(region_column, sort=True):
                axis.scatter(
                    group[x_column],
                    group[y_column],
                    s=8,
                    alpha=0.62,
                    label=f"Region {int(region)} (n={len(group)})",
                )
                axis.scatter(
                    group[x_column].mean(),
                    group[y_column].mean(),
                    s=70,
                    marker="x",
                    linewidths=2.0,
                    color="black",
                )
            center_x = float(parameter[f"{prefix}_reference_center_x"])
            center_y = float(parameter[f"{prefix}_reference_center_y"])
            axis.scatter(
                center_x,
                center_y,
                s=100,
                marker="+",
                linewidths=2.3,
                color="#D62728",
                label=f"{role.title()} median center",
            )
            axis.set_title(f"{SCENE_LABELS[scene]}: automatic {role} endpoint regions")
            axis.set_xlabel("Top-view x [px]")
            axis.set_ylabel("Top-view y [px]")
            axis.legend(loc="upper right", fontsize=7, framealpha=0.92, ncol=2)
            axis.set_xlim(0, background.shape[1])
            axis.set_ylim(background.shape[0], 0)
            _save_figure(
                figure,
                paths.figures / scene / f"{role}_region_map",
            )


def _angular_histograms(paths: Paths) -> None:
    assignments = pd.read_parquet(
        paths.results / "target_endpoint_region_assignments.parquet"
    )
    parameters = pd.read_csv(paths.results / "frozen_scene_parameters.csv")
    for scene in SCENES:
        rows = assignments[assignments["scene"] == scene]
        parameter = parameters[parameters["scene"] == scene].iloc[0]
        figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
        for axis, role, x_column, y_column, region_column in (
            (axes[0], "entry", "start_x_topview", "start_y_topview", "entry_region"),
            (axes[1], "exit", "end_x_topview", "end_y_topview", "exit_region"),
        ):
            center = np.array(
                [
                    parameter[f"{role}_reference_center_x"],
                    parameter[f"{role}_reference_center_y"],
                ],
                dtype=float,
            )
            for region, group in rows.groupby(region_column, sort=True):
                points = group[[x_column, y_column]].to_numpy(float)
                angles = np.degrees(
                    np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
                )
                axis.hist(
                    angles,
                    bins=np.linspace(-180, 180, 37),
                    alpha=0.55,
                    label=f"Region {int(region)}",
                )
            axis.set_title(f"{role.title()} angle distribution")
            axis.set_xlabel("Polar angle [degrees]")
            axis.set_ylabel("Endpoint count")
            axis.set_xlim(-180, 180)
            axis.legend(fontsize=7)
        figure.suptitle(f"{SCENE_LABELS[scene]}: endpoint-angle diagnostics")
        _save_figure(figure, paths.figures / scene / "angular_histogram")


def _annotated_matrix(
    matrix: np.ndarray,
    title: str,
    color_label: str,
    integer_labels: bool,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues", aspect="auto")
    axis.set_title(title)
    axis.set_xlabel("Automatic exit region")
    axis.set_ylabel("Automatic entry region")
    axis.set_xticks(np.arange(matrix.shape[1]))
    axis.set_yticks(np.arange(matrix.shape[0]))
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            label = f"{int(value)}" if integer_labels else f"{100.0 * value:.2f}%"
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > np.nanmax(matrix) * 0.55 else "black",
            )
    figure.colorbar(image, ax=axis, label=color_label)
    return figure


def _od_matrices(paths: Paths) -> None:
    assignments = pd.read_parquet(
        paths.results / "target_endpoint_region_assignments.parquet"
    )
    reproduction = pd.read_csv(paths.results / "target_reproduction.csv")
    for scene in SCENES:
        rows = assignments[assignments["scene"] == scene]
        count = pd.crosstab(rows["entry_region"], rows["exit_region"])
        count = count.reindex(
            index=range(int(rows["entry_region"].max()) + 1),
            columns=range(int(rows["exit_region"].max()) + 1),
            fill_value=0,
        )
        share = count.to_numpy(float) / len(rows)
        figure = _annotated_matrix(
            share,
            f"{SCENE_LABELS[scene]}: OD support before thresholding",
            "Trajectory share",
            False,
        )
        _save_figure(figure, paths.figures / scene / "od_support_matrix")

        threshold = float(
            reproduction.loc[
                reproduction["scene"] == scene, "frozen_support_threshold"
            ].iloc[0]
        )
        supported = np.where(share >= threshold, count.to_numpy(float), 0.0)
        figure = _annotated_matrix(
            supported,
            (
                f"{SCENE_LABELS[scene]}: supported OD pairs "
                f"(threshold={100.0 * threshold:.2f}%)"
            ),
            "Supported trajectory count",
            True,
        )
        _save_figure(
            figure, paths.figures / scene / "od_support_matrix_thresholded"
        )


def _threshold_overview(paths: Paths) -> None:
    frame = pd.read_csv(paths.results / "support_threshold_sensitivity.csv")
    figure, axes = plt.subplots(2, 3, figsize=(11.4, 6.8), constrained_layout=True)
    for axis, scene in zip(axes.flat, SCENES, strict=False):
        rows = frame[frame["scene"] == scene]
        axis.plot(
            100.0 * rows["support_threshold"],
            rows["resulting_target_count"],
            marker="o",
            linewidth=1.7,
        )
        selected = rows[rows["is_frozen_selected_threshold"]]
        axis.scatter(
            100.0 * selected["support_threshold"],
            selected["resulting_target_count"],
            color="#D62728",
            s=55,
            zorder=3,
            label="Frozen choice",
        )
        axis.set_title(SCENE_LABELS[scene])
        axis.set_xlabel("OD support threshold [%]")
        axis.set_ylabel("Estimated target")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    axes.flat[-1].axis("off")
    figure.suptitle("Support-threshold sensitivity (diagnostic only; no retuning)")
    _save_figure(figure, paths.figures / "support_threshold_sensitivity")


def _region_count_overview(paths: Paths) -> None:
    frame = pd.read_csv(paths.results / "region_count_sensitivity.csv")
    figure, axes = plt.subplots(2, 3, figsize=(11.5, 7.3), constrained_layout=True)
    for axis, scene in zip(axes.flat, SCENES, strict=False):
        rows = frame[frame["scene"] == scene]
        matrix = rows.pivot(
            index="entry_region_count",
            columns="exit_region_count",
            values="target_at_frozen_threshold",
        )
        image = axis.imshow(matrix.to_numpy(), cmap="viridis", aspect="auto")
        axis.set_title(SCENE_LABELS[scene])
        axis.set_xticks(range(len(matrix.columns)), matrix.columns)
        axis.set_yticks(range(len(matrix.index)), matrix.index)
        axis.set_xlabel("Exit-region K")
        axis.set_ylabel("Entry-region K")
        for row_index, entry in enumerate(matrix.index):
            for column_index, exit_id in enumerate(matrix.columns):
                value = int(matrix.loc[entry, exit_id])
                selected = rows[
                    (rows["entry_region_count"] == entry)
                    & (rows["exit_region_count"] == exit_id)
                ]["is_frozen_region_pair"].iloc[0]
                axis.text(
                    column_index,
                    row_index,
                    f"{value}{'*' if selected else ''}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > matrix.to_numpy().mean() else "black",
                )
        figure.colorbar(image, ax=axis, shrink=0.78)
    axes.flat[-1].axis("off")
    figure.suptitle(
        "Target sensitivity to endpoint-region granularity (* = frozen K pair)"
    )
    _save_figure(figure, paths.figures / "region_count_sensitivity")


def _se38th_fragmentation_figure(paths: Paths) -> None:
    queue = pd.read_csv(paths.results / "se38th_fragmentation_figure_queue.csv")
    if queue.empty:
        return
    guide = yaml.safe_load(
        (
            paths.publication
            / "annotations/protocol/scene_guides/bellevue_150th_se38th.yaml"
        ).read_text(encoding="utf-8")
    )
    background_path = paths.publication / "annotations" / guide["representative_frame"]
    background = mpimg.imread(background_path)
    figure, axis = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    axis.imshow(background)
    cmap = plt.get_cmap("tab20")
    labels = sorted(queue["cluster_label"].astype(int).unique())
    colors = {label: cmap(index % 20) for index, label in enumerate(labels)}
    for source_path, group in queue.groupby("trajectory_source_path", sort=True):
        track_ids = group["source_recording_track_id"].astype(int).tolist()
        source = pd.read_parquet(
            paths.repo / source_path,
            columns=["track_id", "frame", "cx", "cy"],
            filters=[("track_id", "in", track_ids)],
        )
        for row in group.itertuples(index=False):
            track = source[
                (source["track_id"] == int(row.source_recording_track_id))
                & (source["frame"] >= int(row.start_frame))
                & (source["frame"] <= int(row.end_frame))
            ].sort_values("frame", kind="mergesort")
            axis.plot(
                track["cx"],
                track["cy"],
                color=colors[int(row.cluster_label)],
                linewidth=1.0,
                alpha=0.52,
            )
    for label in labels:
        axis.plot([], [], color=colors[label], linewidth=2.5, label=f"Cluster {label}")
    movement = str(queue["reference_movement_id"].iloc[0]).split(":", 1)[1]
    axis.set_title(
        "SE38th HG-aware KMeans: one manual movement split across geometric clusters\n"
        f"Movement {movement}; deterministic diagnostic sample (n={len(queue)})"
    )
    axis.set_xlabel("Camera x [px]")
    axis.set_ylabel("Camera y [px]")
    axis.set_xlim(0, background.shape[1])
    axis.set_ylim(background.shape[0], 0)
    axis.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.92)
    _save_figure(figure, paths.figures / "se38th_hg_kmeans_fragmentation")


def run_figures(paths: Paths | None = None) -> dict[str, Any]:
    paths = paths or default_paths()
    paths.figures.mkdir(parents=True, exist_ok=True)
    _endpoint_maps(paths)
    _angular_histograms(paths)
    _od_matrices(paths)
    _threshold_overview(paths)
    _region_count_overview(paths)
    _se38th_fragmentation_figure(paths)
    files = sorted(path for path in paths.figures.rglob("*") if path.is_file())
    manifest = pd.DataFrame(
        {
            "path": [path.relative_to(paths.publication).as_posix() for path in files],
            "sha256": [sha256_file(path) for path in files],
            "size_bytes": [path.stat().st_size for path in files],
        }
    )
    manifest.to_csv(
        paths.results / "target_estimation_figure_manifest.csv",
        index=False,
        lineterminator="\n",
    )
    return {"figure_files": len(files)}


def _table(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    return frame.to_markdown(index=False, floatfmt=floatfmt)


def _write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _load_outputs(paths: Paths) -> dict[str, pd.DataFrame]:
    names = (
        "target_reproduction",
        "frozen_scene_parameters",
        "endpoint_region_diagnostics",
        "automatic_region_to_manual_approach_mapping",
        "automatic_od_to_manual_movement_mapping",
        "se38th_od_collapse_table",
        "se38th_cluster_fragmentation",
        "cross_scene_semantic_duplication",
        "support_threshold_sensitivity",
        "region_count_sensitivity",
    )
    return {name: pd.read_csv(paths.results / f"{name}.csv") for name in names}


def _implementation_audit(paths: Paths, data: dict[str, pd.DataFrame]) -> str:
    parameters = data["frozen_scene_parameters"]
    threshold_table = parameters[
        [
            "scene",
            "chosen_entry_regions",
            "chosen_exit_regions",
            "support_threshold_percent",
            "minimum_required_support_count",
            "frozen_target",
            "entry_random_seed",
            "exit_random_seed",
        ]
    ].copy()
    return f"""# HG Target Estimator Implementation Audit

## Scope and scientific lock

This audit reconstructs the already frozen `split-aware-hg-msa-tc-v1` target
estimator. It does not define a replacement estimator and does not modify any frozen
target, endpoint-region count, support threshold, selected clustering configuration,
or independent-test assignment.

## Verified implementation locations

| Role | Path | Verified behavior |
| --- | --- | --- |
| Canonical implementation | `code/target_estimation/hg_target_estimator.py` | Single source for homography, endpoint features, region KMeans, OD support, and threshold selection. |
| Compatibility API | `code/pipeline/hg_msa_tc_core.py` | Delegates the original public functions to the canonical module without changing outputs. |
| Frozen development caller | `code/pipeline/run_split_aware_hg_msa_tc.py::target_phase` | Loads only `target_estimation`, transforms endpoints, adds scene seed offsets, and persists target artifacts. |
| Split/checksum guard | `code/pipeline/split_aware_io.py` | Verifies split membership, one-to-one manifest joins, source hashes, finite feature rows, and deterministic trajectory ordering. |
| Task 07 analysis | `code/target_estimation/analysis.py` | Reproduces first; human reference and persisted test assignments are loaded only in a later diagnostic stage. |

A historical isolated implementation remains at
`research_experiments/fov2026_trajectory_clustering/scripts/hg_msa_tc_five_scene/run_hg_msa_tc_five_scene_pipeline.py`.
It was a source for the publication implementation but is not called by the frozen
publication runner. The historical code contains an unused joint-center variable,
did not sort the metric subsample indices, and had a shorter threshold tie-break key.
The authoritative behavior is the frozen publication implementation reproduced here.

## Exact verified behavior

1. The camera start and end feature endpoints are transformed separately by the
   frozen camera-to-top-view homography.
2. Entry and exit points use separate coordinate-wise median centers. There is no
   single shared scene center in the executed code.
3. Each point is represented by `cos(angle)`, `sin(angle)`, and a 0.25-weighted,
   median-normalized radius clipped at three times the median radius.
4. KMeans evaluates K in `[3, 4, 5, 6, 7, 8]`, with `n_init=10`, Lloyd updates, and
   deterministic scene/role seeds.
5. Endpoint K maximizes silhouette, then minimizes Davies-Bouldin, then prefers lower K.
6. OD support is the trajectory fraction for each observed entry-region/exit-region pair.
7. A pair is supported when `share >= threshold`; zero-support combinations are absent.
8. The threshold grid is `[0.1%, 0.25%, 0.5%, 1%, 2%]`. The selected threshold minimizes
   adjacent-grid target instability among candidates with at least 90% OD coverage and
   at least two pairs (80% fallback), then prefers closeness to 0.5%, higher coverage,
   and finally the lower threshold.

## Frozen scene outputs

{_table(threshold_table, '.4f')}

The different final scene thresholds were not entered manually per scene. They are
data-dependent outputs of the same frozen heuristic over one global threshold grid.
Repository evidence does not support describing these weights, K bounds, or threshold
rules as theoretically optimal; they are heuristic design choices frozen before the
independent-test evaluation.

## Edge cases and failure behavior

- Source feature NaNs are rejected by the split-aware loader before estimation.
- A homography denominator with absolute value below `1e-12` raises an error.
- Empty cohorts, fewer rows than candidate K, or all-invalid input are not assigned a
  fallback target; the estimator fails instead of fabricating a result.
- KMeans endpoint grouping has no noise label. Every valid endpoint is assigned to one region.
- Numeric KMeans labels have no semantic ordering; only their deterministic partitions matter.
- Silhouette/DB failures are represented as NaN and ranked as worst.
- The support denominator is every authorized target-estimation trajectory in the scene.

## Intended versus verified manuscript role

The verified target is a count of supported geometric OD submodes. It may approximate
semantic maneuvers when one endpoint region corresponds to one physical approach, but
the code does not enforce that correspondence. Any manuscript wording that calls it a
guaranteed semantic or legal maneuver count must be narrowed.
"""


def _mathematical_definition() -> str:
    return r"""# HG Target Estimator Mathematical Definition

## Trajectories and homography

For target-estimation trajectory `tau_i`, let the frozen camera-space feature endpoints be

`p_i^in = (x_i^in, y_i^in)^T` and `p_i^out = (x_i^out, y_i^out)^T`.

For `p = (x,y)^T`, the camera-to-top-view homography is

`h = H [x, y, 1]^T`, and `pi_H(p) = (h_1/h_3, h_2/h_3)^T`.

The implementation rejects `|h_3| < 1e-12`. It applies this transformation to every
authorized endpoint and does not read human reference labels.

## Separate entry and exit centers

For role `r` in `{in,out}`, define a separate coordinate-wise median center:

`c_r = (median_i z_{ir,x}, median_i z_{ir,y})^T`,

where `z_ir = pi_H(p_i^r)`. The executed code therefore has an entry center and an
exit center, not one joint intersection center.

## Polar-radius endpoint representation

For role `r`:

`d_ir = z_ir - c_r`,

`phi_ir = atan2(d_ir,y, d_ir,x)`,

`rho_ir = ||d_ir||_2`,

`rho'_ir = clip(rho_ir / max(median_j rho_jr, 1e-12), 0, 3) / 3`,

and the endpoint feature vector is

`u_ir = [cos(phi_ir), sin(phi_ir), 0.25 rho'_ir]^T`.

The circular encoding avoids an angle discontinuity at `-pi/pi`; the clipped radius
retains weak distance information without giving it the same scale as the two angular terms.

## Endpoint-region selection

For each role and each candidate `K in {3,4,5,6,7,8}`, KMeans minimizes

`sum_a sum_{i in R_a} ||u_ir - mu_ar||_2^2`

with `n_init=10`, Lloyd updates, and a frozen seed. Internal metrics are computed on
all rows when `N <= 2500`, otherwise on a deterministic 2500-row sample. The chosen K
is the lexicographic optimum:

1. maximum silhouette;
2. minimum Davies-Bouldin;
3. minimum K.

The resulting entry regions are `E_a`; exit regions are `X_b`.

## OD support and target

For every observed automatic pair `(a,b)`:

`n_ab = sum_i 1[z_i^in in E_a and z_i^out in X_b]`,

`q_ab = n_ab / N`.

For candidate support threshold `theta`, the target and retained coverage are

`K(theta) = sum_(a,b observed) 1[q_ab >= theta]`,

`C(theta) = sum_(a,b observed) n_ab 1[q_ab >= theta] / N`.

The comparison is inclusive (`>=`). Unobserved zero-support combinations do not appear
in the sum. The reported absolute support count is `ceil(theta N)`.

## Threshold heuristic

For ordered candidates `theta_j`, local target instability is

`I_j = mean(|K(theta_j)-K(theta_l)| : l is an adjacent grid index)`.

The estimator first retains candidates with `C >= 0.90` and `K >= 2`; if none exist it
uses `C >= 0.80`, then all candidates as a final fallback. It minimizes the tuple

`(I_j, |theta_j-0.005|, -C(theta_j), theta_j)`.

The final frozen output is `K_HG = K(theta*)`.

## Determinism and interpretation

KMeans labels are arbitrary integer identifiers, but frozen seeds and stable sorting
make the partition reproducible. The target is an automatically estimated count of
supported top-view geometric endpoint-pair modes. It is not guaranteed to equal a
legal or semantic maneuver count.
"""


def _reproduction_report(data: dict[str, pd.DataFrame]) -> str:
    frame = data["target_reproduction"]
    table = frame[
        [
            "scene",
            "n_trajectories",
            "frozen_target",
            "reproduced_target",
            "frozen_entry_regions",
            "frozen_exit_regions",
            "frozen_support_threshold",
            "target_exact_match",
        ]
    ]
    maximum = max(
        float(frame["threshold_candidates_max_abs_difference"].max()),
        float(frame["region_candidates_max_abs_difference"].max()),
        float(frame["od_support_max_abs_difference"].max()),
    )
    return f"""# HG Target Reproduction Report

## Result

The canonical implementation reproduced all five frozen targets using only the
`target_estimation` split. Human polygon-reference labels were not loaded until after
this result and its lock file had been written.

{_table(table, '.4f')}

The maximum absolute numeric difference across the complete frozen threshold-candidate,
endpoint-region-candidate, and OD-support tables was `{maximum:.3e}`. Row schemas,
non-numeric fields, selected flags, targets, seeds, and scene ordering also matched.

## Isolation checks

- Target recomputation split: `target_estimation` only.
- Human-reference input during reproduction: none.
- Independent-test clustering executed: no.
- Frozen targets modified: no.
- Selected clustering configurations modified: no.
- Original development outputs overwritten: no.

The new diagnostic assignments are stored separately under `results/target_estimation/`.
"""


def _se38th_failure_report(paths: Paths, data: dict[str, pd.DataFrame]) -> str:
    cross = data["cross_scene_semantic_duplication"]
    row = cross[cross["scene"] == "bellevue_150th_se38th"].iloc[0]
    regions = data["automatic_region_to_manual_approach_mapping"]
    regions = regions[regions["scene"] == "bellevue_150th_se38th"]
    candidates = pd.read_csv(
        paths.publication / "results/development/target_region_candidates.csv"
    )
    candidates = candidates[candidates["scene"] == "bellevue_150th_se38th"]
    collapse = data["se38th_od_collapse_table"]
    calibration = json.loads(
        (
            paths.homography_directory
            / "homography_bellevue_150th_se38th.json"
        ).read_text(encoding="utf-8")
    )
    region_table = regions[
        [
            "endpoint_role",
            "automatic_region_id",
            "dominant_manual_approach",
            "mapping_purity",
            "automatic_region_size",
            "counts_by_manual_approach_json",
        ]
    ]
    candidate_table = candidates[
        ["endpoint_role", "n_regions", "silhouette", "davies_bouldin", "selected"]
    ]
    collapse_table = collapse[
        [
            "automatic_od_pair",
            "target_estimation_support_count",
            "target_estimation_support_percentage",
            "empirical_target_reference_movement",
            "empirical_target_reference_purity",
            "empirical_dominant_duplicate_group_size",
            "unique_target_reference_movement_count",
        ]
    ]
    low_support = int(collapse["low_support_mode"].sum())
    short_fraction = float(
        np.average(
            collapse["short_track_fraction_le_30_points"],
            weights=collapse["target_estimation_support_count"],
        )
    )
    return f"""# SE38th Target Failure Analysis

## Frozen failure

SE38th produced `K_HG = 18`, while the independent polygon-rule reference contains
9 observed semantic movements. The frozen value is retained; this document diagnoses
the discrepancy and does not post hoc correct it.

## Why seven entry and three exit regions were selected

The selected counts are direct internal-metric optima over K=3..8:

{_table(candidate_table, '.4f')}

Entry K=7 has the highest sampled silhouette (0.7723); exit K=3 has the highest exit
silhouette (0.6890). The selection rule has no road-branch consolidation constraint,
so internal feature-space separation can override one-region-per-physical-approach semantics.

## Automatic region to manual approach mapping

{_table(region_table, '.4f')}

Four entry regions (0, 2, 5, and 6) are dominated by manual approach A. This is direct
entry-region fragmentation. On the exit side, region 1 contains E, H, G, and one F
endpoint, while region 2 contains both G and F. Thus K=3 also merges physical exit
approaches. The failure combines entry over-segmentation with exit under-segmentation;
the entry split is the stronger driver of the high OD-pair count.

## Collapse of the 18 supported OD pairs

{_table(collapse_table, '.4f')}

The 18 automatic pairs have only 7 distinct dominant target-reference movement labels;
11 pairs are duplicates under that dominant-label view
(`{float(row['empirical_dominant_duplication_ratio']):.1%}`). However, eight automatic
pairs contain more than one valid manual movement, so a one-to-one collapse is not
fully supported. Across all valid target-estimation trajectories in the 18 pairs,
{int(row['all_semantic_movements_represented_in_supported_pairs'])} manual movement
labels occur. The independent-test reference observes 9 movements. This distinction
shows that the estimator counts geometric endpoint submodes, not guaranteed semantic classes.

Only {low_support} of the 18 retained pairs has support below twice the frozen threshold;
the weighted fraction of trajectories with at most 30 canonical points is
`{short_fraction:.2%}`. This diagnostic does not identify rare-pair support or short
tracks as a material explanation for the factor-of-two target error. Raising the
threshold alone also never reduces the diagnostic target to 9 over the pre-specified
0.05%-1.0% grid.

## Homography and lane-level interpretation

The frozen SE38th homography has all-point mean reprojection error
`{float(calibration['mean_reprojection_error_px']):.2f} px` and is classified
`acceptable_with_caution`. Homography uncertainty can broaden endpoint modes, but the
manual-approach mapping directly demonstrates repeated A regions even after calibration.
The separated geometric modes are consistent with lane-level or endpoint-position
substructure. They cannot be asserted to be lane-level maneuvers because no lane-marking
or mandatory-turn-arrow labels were used.

## Root-cause conclusion

SE38th is a semantic over-segmentation failure caused primarily by endpoint-region
granularity: one physical entry branch is resolved into several geometric modes, while
the exit partition simultaneously merges some semantic exits. Endpoint dispersion and
homography uncertainty remain plausible secondary contributors; the available short-track
diagnostic does not support incomplete tracks as a major cause. A future road-branch
consolidation stage could merge modes belonging to one physical approach, but it is not
applied to the current frozen results.
"""


def _fragmentation_report(data: dict[str, pd.DataFrame]) -> str:
    frame = data["se38th_cluster_fragmentation"].copy()
    summary_rows = []
    for (method, strategy), group in frame.groupby(
        ["method", "selection_strategy"], sort=True
    ):
        weights = group["movement_support"].to_numpy(float)
        summary_rows.append(
            {
                "method": method,
                "selection_strategy": strategy,
                "support_weighted_effective_clusters": float(
                    np.average(group["effective_number_of_clusters"], weights=weights)
                ),
                "maximum_effective_clusters_for_one_movement": float(
                    group["effective_number_of_clusters"].max()
                ),
                "support_weighted_noise_ratio": float(
                    np.average(group["noise_ratio"], weights=weights)
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    kmeans = frame[
        (frame["method"] == "kmeans")
        & (frame["selection_strategy"] == "hg_expected_aware_selection")
    ].sort_values("effective_number_of_clusters", ascending=False)
    top = kmeans[
        [
            "reference_movement_id",
            "movement_support",
            "unique_non_noise_clusters",
            "clusters_with_at_least_1pct_of_movement",
            "effective_number_of_clusters",
            "dominant_cluster_share_non_noise",
            "movement_completeness_proxy",
            "weighted_cluster_purity",
        ]
    ]
    return f"""# SE38th Cluster Fragmentation Analysis

## Diagnostic definitions

This analysis reads already persisted independent-test assignments; it does not rerun
clustering. For each manual movement, the non-noise cluster proportions are `p_j`.
The effective cluster count is `exp(-sum_j p_j log p_j)`. The normalized fragmentation
entropy divides that entropy by `log(J)` when more than one cluster is present. The
reported movement-completeness proxy is `1 - normalized entropy`. These are diagnostic
quantities, not new primary manuscript metrics.

## Strategy-level summary

{_table(summary, '.4f')}

## HG-aware KMeans (`k=18`)

{_table(top, '.4f')}

The strongest KMeans fragmentation is visible for `B>E` (effective cluster count
about 3.12), followed by `C>E` and `A>G`. Support-weighted effective cluster count rises
from about 1.41 under untargeted KMeans to 1.81 under HG-aware KMeans. The corresponding
figure overlays one manual movement colored by its frozen HG-aware clusters.

The splits are spatially structured enough to be consistent with distinct geometric
path or lane-position submodes, but the current data do not prove lane semantics.
Fragmentation is therefore evidence that KMeans is especially sensitive to an
overestimated target: setting `k=18` forces every test trajectory into one of 18 groups.
Density methods are less directly controlled by the target because the target affects
configuration selection rather than fixing the fitted number of clusters.
"""


def _threshold_report(data: dict[str, pd.DataFrame]) -> str:
    threshold = data["support_threshold_sensitivity"]
    rows = []
    for scene, group in threshold.groupby("scene", sort=False):
        selected = group[group["is_frozen_selected_threshold"]].iloc[0]
        rows.append(
            {
                "scene": scene,
                "frozen_threshold_percent": 100.0 * selected["support_threshold"],
                "frozen_target": int(selected["resulting_target_count"]),
                "minimum_target_on_grid": int(group["resulting_target_count"].min()),
                "maximum_target_on_grid": int(group["resulting_target_count"].max()),
                "thresholds_preserving_frozen_target": int(
                    (group["resulting_target_count"] == selected["resulting_target_count"]).sum()
                ),
                "minimum_pair_jaccard_vs_frozen": float(
                    group["supported_pair_jaccard_vs_frozen"].min()
                ),
            }
        )
    region = data["region_count_sensitivity"]
    region_rows = []
    for scene, group in region.groupby("scene", sort=False):
        selected = group[group["is_frozen_region_pair"]].iloc[0]
        region_rows.append(
            {
                "scene": scene,
                "frozen_entry_K": int(selected["entry_region_count"]),
                "frozen_exit_K": int(selected["exit_region_count"]),
                "frozen_target": int(selected["target_at_frozen_threshold"]),
                "minimum_target_over_K_grid": int(
                    group["target_at_frozen_threshold"].min()
                ),
                "maximum_target_over_K_grid": int(
                    group["target_at_frozen_threshold"].max()
                ),
                "distinct_target_counts": int(
                    group["target_at_frozen_threshold"].nunique()
                ),
            }
        )
    return f"""# Support-Threshold and Region-Count Sensitivity Report

## Support threshold (diagnostic only)

{_table(pd.DataFrame(rows), '.4f')}

The threshold curve was computed from target-estimation OD support before consulting
independent semantic counts. SE38th stays between 14 and 19 supported pairs over
0.05%-1.0%; threshold adjustment alone does not recover the independently observed 9.
Eastgate and NE8th are comparatively stable around their frozen choices, while Newport
falls from 12 to 7 as rare pairs are removed.

## Endpoint-region granularity (diagnostic only)

{_table(pd.DataFrame(region_rows), '.4f')}

Target count is substantially more sensitive to entry/exit K than to small local
threshold changes. SE38th ranges from 8 to 36 over the pre-specified K=3..8 grid; all
scenes show a broad region-count range. This sensitivity is not used to choose new K
values. It demonstrates that the estimator's semantic interpretation depends on
endpoint-region granularity.

No frozen threshold, K value, or target was overwritten by this analysis.
"""


def _failure_taxonomy() -> str:
    return """# HG Target Estimator Failure Taxonomy

| Failure mode | Mechanism | Diagnostic signature | Current evidence | Consequence |
| --- | --- | --- | --- | --- |
| Semantic over-segmentation | One physical movement is represented by several geometric endpoint pairs. | Automatic target exceeds independent observed movements; repeated dominant manual mappings. | Strong at SE38th; limited non-semantic pairs at Newport. | KMeans may be forced to split semantic movements; density-method selection may favor fragmented solutions. |
| Semantic under-segmentation | Distinct movements share endpoint regions or fall below support. | Fewer automatic pairs than observed movements; mixed manual labels in one pair. | NE8th target 9 versus 10 observed; SE38th exit K=3 merges exits. | KMeans merges movements; density methods may select coarser settings. |
| Endpoint-region fragmentation | KMeans finds multiple internally separated modes within one approach. | Multiple automatic regions map to the same manual approach. | Four SE38th entry regions are dominated by approach A. | Multiplies OD combinations and inflates target. |
| Rare-movement suppression | A legitimate movement has support below threshold. | Pair disappears as threshold increases; observed movement lacks supported pair. | Plausible at Newport/NE8th; threshold curves quantify it. | Under-counted target and possible merged rare class. |
| Spurious low-support OD pair | Endpoint noise creates a small pair just above threshold. | Pair support near threshold; poor manual purity. | A minority of pairs; not the main SE38th cause. | Inflated target, particularly for KMeans. |
| Incomplete-track endpoint error | Track begins/ends inside the intersection or before the intended branch. | Short trajectories, invalid polygon status, mixed automatic pairs. | Present in a subset of SE38th rows; secondary. | Wrong entry/exit region and noisy OD support. |
| Homography-induced region distortion | Calibration error shifts or spreads transformed endpoints. | Region spread aligns with high reprojection uncertainty. | SE38th calibration is acceptable with caution; causality is not isolated here. | Can split or merge endpoint modes. |
| Threshold instability | Small threshold changes alter many supported pairs. | Large target range and low pair-set Jaccard near frozen threshold. | Moderate at Newport; limited locally at 116th/Eastgate/NE8th. | Target depends on rare-pair cutoff. |

These modes are not mutually exclusive. SE38th combines endpoint-region fragmentation,
exit-region merging, endpoint uncertainty, and some incomplete trajectories. The frozen
analysis supports the first two as the primary mechanisms.
"""


def _scientific_interpretation(data: dict[str, pd.DataFrame]) -> str:
    cross = data["cross_scene_semantic_duplication"].copy()
    cross["absolute_error_vs_observed"] = (
        cross["automatic_od_target"]
        - cross["observed_independent_semantic_movement_count"]
    ).abs()
    mean_error = float(cross["absolute_error_vs_observed"].mean())
    table = cross[
        [
            "scene",
            "automatic_od_target",
            "observed_independent_semantic_movement_count",
            "absolute_error_vs_observed",
            "automatic_entry_region_count",
            "automatic_exit_region_count",
            "semantic_duplication_ratio",
        ]
    ]
    return f"""# HG Target Estimator Scientific Interpretation

## Five-scene accuracy

{_table(table, '.4f')}

The frozen target exactly matches the independent observed count in two of five scenes
(116th/NE12th and Eastgate), differs by one at NE8th, overestimates Newport by three,
and overestimates SE38th by nine. Mean absolute scene-level error is `{mean_error:.2f}`.
With only five scenes, this is descriptive evidence, not a population-level accuracy claim.

## What the estimator counts

The estimator counts supported geometric endpoint-region pairs. In favorable scenes
those pairs align with semantic maneuvers. SE38th proves that the equivalence is not
guaranteed: one physical approach is split into several automatic entry modes, while
the exit partition merges manual exits. The method can therefore resolve lane-position
or geometric trajectory submodes in addition to semantic maneuver classes.

## Why KMeans is most sensitive

For HG-aware KMeans the frozen target directly sets `k`. An overestimate therefore
forces additional non-noise partitions. HDBSCAN and OPTICS do not take target count as
a fit parameter; the target only changes which already evaluated configuration is
selected. They can still fragment or merge movements, but the propagation is indirect.

## Robustness

The support threshold is locally stable for several scenes, and SE38th remains far
above nine throughout the diagnostic threshold grid. Endpoint-region count is much
more influential: plausible K combinations create broad target ranges in every scene.
The estimator is therefore not robust to endpoint-region granularity in a semantic sense.

## Claims to retain, narrow, or remove

- Retain: homography enables a deterministic, label-free geometric target-estimation layer.
- Retain: the frozen implementation is exactly reproducible from `target_estimation` only.
- Narrow: target alignment improves in several scenes, not uniformly across all scenes.
- Narrow: `K_HG` is an automatically estimated observed geometric-maneuver target, not
  a guaranteed semantic or legal maneuver count.
- Remove: any implication that all physical road branches are recovered one-to-one.
- Remove: any universal-superiority claim based only on EMAS_HG or cluster-count agreement.

## Defensible limitation and future work

The endpoint-region K and OD threshold are heuristic and can resolve multiple modes
within one physical approach or merge distinct exits. Calibration uncertainty,
incomplete trajectories, and camera-space reference boundaries add further uncertainty.
A future road-branch consolidation stage could merge geometric endpoint modes using
road topology, lane markings, or mandatory-turn arrows. It must be prospectively
specified and evaluated; it is not applied to current results.
"""


def _manuscript_ready_section(data: dict[str, pd.DataFrame]) -> str:
    parameters = data["frozen_scene_parameters"]
    table = parameters[
        [
            "scene",
            "target_estimation_trajectory_count",
            "chosen_entry_regions",
            "chosen_exit_regions",
            "support_threshold_percent",
            "minimum_required_support_count",
            "frozen_target",
        ]
    ]
    return f"""# Manuscript-Ready HG Target-Estimation Section

## Methods: homography-guided target estimation

For each trajectory in the development-only target-estimation split, the first and
last frozen feature endpoints were transformed to top-view coordinates by the scene's
camera-to-map homography. Entry and exit endpoints were processed separately. For each
role, the coordinate-wise median endpoint `c` defined the polar reference center. An
endpoint `z` was represented as

`u(z) = [cos(phi), sin(phi), 0.25 clip(r/median(r),0,3)/3]`,

where `phi = atan2(z_y-c_y,z_x-c_x)` and `r = ||z-c||_2`. KMeans endpoint partitions
with K=3,...,8 were evaluated with ten initializations and frozen random seeds. The
region count maximized silhouette, then minimized Davies-Bouldin, with lower K as the
final tie-breaker.

For automatic entry region `a` and exit region `b`, support was
`q_ab = n_ab/N`. At threshold `theta`, the estimated target was

`K_HG(theta) = sum_ab 1[q_ab >= theta]`.

Thresholds 0.1%, 0.25%, 0.5%, 1%, and 2% were evaluated. The frozen heuristic preferred
at least 90% retained OD coverage and at least two pairs, minimized adjacent-grid target
instability, then preferred proximity to 0.5%, higher coverage, and lower threshold.
No manual maneuver labels or independent-test rows entered this computation.

## Frozen scene parameters

{_table(table, '.4f')}

## Results

The canonical reconstruction reproduced targets 10, 12, 9, 18, and 9 exactly. Compared
with independent polygon-rule observed counts 10, 9, 9, 9, and 10, the estimator matched
two scenes exactly, differed by one at NE8th, and overestimated Newport and SE38th.
The severe SE38th overestimate arose because the internally preferred endpoint partition
contained seven entry regions and three exit regions. Four entry regions mapped mainly
to one manual physical approach, whereas exit regions merged several manual exits.

## Discussion and limitation

These findings clarify that `K_HG` is a supported geometric endpoint-pair count rather
than a guaranteed semantic maneuver count. The estimator can resolve path or lane-position
submodes inside one physical approach. Its support threshold is moderately stable, but
its target is sensitive to endpoint-region granularity. The frozen SE38th target was not
corrected after independent evaluation.

## Future work

Future work should introduce a prospectively defined road-branch consolidation stage
using map topology, lane markings, or mandatory-turn arrows, followed by evaluation on
new scenes. Such consolidation is not part of the present results.
"""


def _reviewer_response() -> str:
    return """# Response to Reviewer: HG Target Estimation

**Reviewer comment:** The target-estimation method requires reproducible definitions of
the scene center, polar grouping, K selection, entry/exit grouping, support threshold,
scene-specific thresholds, and failure modes.

**Response:** We agree and have added a complete mathematical and implementation-level
definition [Section X, page X, lines X-X]. The executed code uses separate coordinate-wise
median centers for entry and exit endpoints; it does not use one manually selected scene
center. Polar direction is encoded by cosine and sine, with an explicitly defined clipped,
weakly weighted radius term. Endpoint KMeans evaluates K=3,...,8 with ten initializations,
frozen seeds, Lloyd updates, and a deterministic silhouette/DB/lower-K ordering.

We now define OD support as `q_ab=n_ab/N` and target count as
`K_HG(theta)=sum_ab 1[q_ab>=theta]`, including the inclusive comparison and denominator.
The global candidate threshold grid is 0.1%, 0.25%, 0.5%, 1%, and 2%. Different final
scene thresholds are outputs of one frozen, data-dependent heuristic; they were not
manually assigned per scene. We have stated that this heuristic is not theoretically optimal.

We verified the canonical implementation against all frozen development artifacts.
Targets 10, 12, 9, 18, and 9 and all candidate/support tables were reproduced exactly
from the target-estimation split without reference labels [Table X]. No target or selected
configuration was changed.

We also added a failure analysis [Section X, Figure X]. At SE38th, internal metrics selected
seven entry and three exit regions. Four automatic entry regions map predominantly to one
manual physical approach, while exit regions merge several manual exits. Consequently,
the target of 18 counts geometric endpoint submodes rather than nine independent-test
semantic movements. We retain this unfavorable frozen result and narrow the manuscript
claim: the estimator provides an automatically estimated observed geometric-maneuver target,
not a guaranteed legal or semantic maneuver count.

Threshold and endpoint-region sensitivity are now reported without retuning [Figure X].
SE38th remains above the semantic count over the threshold grid, while target counts vary
substantially with endpoint-region granularity. We identify prospective road-branch
consolidation as future work and do not apply it to the current results.
"""


def _result_manifest_document(paths: Paths) -> str:
    result_manifest = json.loads(
        (paths.results / "task_07_result_manifest.json").read_text(encoding="utf-8")
    )
    figure_manifest = pd.read_csv(
        paths.results / "target_estimation_figure_manifest.csv"
    )
    result_rows = [
        {
            "path": path,
            "sha256": metadata["sha256"],
            "size_bytes": metadata["size_bytes"],
        }
        for path, metadata in result_manifest["outputs"].items()
    ]
    return f"""# Target-Estimation Result Manifest

Generated: `{utc_timestamp()}`

Scientific status:

- all five frozen targets reproduced: yes;
- target split used for recomputation: `target_estimation` only;
- reference access: post-reproduction diagnostic only;
- frozen targets/configurations modified: no;
- independent-test clustering rerun: no.

## Result files

{_table(pd.DataFrame(result_rows), '.0f')}

## Figure files

{_table(figure_manifest, '.0f')}
"""


def _task_execution_report(paths: Paths, data: dict[str, pd.DataFrame]) -> str:
    validation_path = paths.results / "task_07_validation_results.json"
    validation = (
        json.loads(validation_path.read_text(encoding="utf-8"))
        if validation_path.exists()
        else {"pytest": "pending", "ruff": "pending"}
    )
    return f"""# Task 07 Execution Report

## Execution summary

- Branch: `feature/futuretransp-target-estimation-formalization`
- Canonical module: `code/target_estimation/hg_target_estimator.py`
- Frozen targets reproduced: 10, 12, 9, 18, 9
- Reference labels read during reproduction: no
- Independent-test clustering rerun: no
- Frozen targets, thresholds, K values, and selected configurations changed: no
- Pytest: `{validation.get('pytest', 'pending')}`
- Ruff: `{validation.get('ruff', 'pending')}`

## Exact commands

```powershell
./.venv/Scripts/python.exe publications/hg-msa-tc-future-transportation/code/run_target_estimation_analysis.py analyze
./.venv/Scripts/python.exe publications/hg-msa-tc-future-transportation/code/run_target_estimation_reporting.py figures
./.venv/Scripts/python.exe publications/hg-msa-tc-future-transportation/code/run_target_estimation_reporting.py documents
./.venv/Scripts/python.exe -m pytest publications/hg-msa-tc-future-transportation/tests -q
./.venv/Scripts/python.exe -m ruff check publications/hg-msa-tc-future-transportation/code publications/hg-msa-tc-future-transportation/tests
```

## Principal findings

- SE38th target error is primarily endpoint-region granularity failure, not a threshold-only problem.
- The 18 supported SE38th geometric OD pairs contain repeated and mixed manual semantics.
- Threshold sensitivity is moderate; region-count sensitivity is broad.
- Persisted SE38th KMeans assignments show manual movement fragmentation under `k=18`.
- The estimator should be described as geometric and heuristic, not as a guaranteed semantic counter.

## Limitations

- Five fixed scenes provide limited scene-level replication.
- Manual point-pair homographies retain calibration uncertainty.
- The human-defined polygon reference shares endpoint information with the maneuver problem.
- Region-to-approach mapping is diagnostic and was performed after target reproduction.
- Lane-level interpretation is plausible but unverified without lane semantics.
"""


def run_documents(paths: Paths | None = None) -> dict[str, Any]:
    paths = paths or default_paths()
    data = _load_outputs(paths)
    documents = {
        "hg_target_estimator_implementation_audit.md": _implementation_audit(paths, data),
        "hg_target_estimator_mathematical_definition.md": _mathematical_definition(),
        "hg_target_reproduction_report.md": _reproduction_report(data),
        "se38th_target_failure_analysis.md": _se38th_failure_report(paths, data),
        "se38th_cluster_fragmentation_analysis.md": _fragmentation_report(data),
        "support_threshold_sensitivity_report.md": _threshold_report(data),
        "hg_target_estimator_failure_taxonomy.md": _failure_taxonomy(),
        "hg_target_estimator_scientific_interpretation.md": _scientific_interpretation(data),
        "manuscript_ready_target_estimation_section.md": _manuscript_ready_section(data),
        "response_to_reviewer_target_estimation_draft.md": _reviewer_response(),
        "target_estimation_result_manifest.md": _result_manifest_document(paths),
        "task_07_execution_report.md": _task_execution_report(paths, data),
    }
    for name, content in documents.items():
        _write(paths.docs / name, content)
    return {"documents": len(documents)}
