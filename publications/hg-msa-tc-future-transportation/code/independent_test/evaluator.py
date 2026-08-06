"""Separate evaluation of persisted clusters against polygon-rule references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
    v_measure_score,
)

from pipeline import hg_msa_tc_core as core
from pipeline import split_aware_io as protocol_io

from .protocol import METHODS, SCENES, STRATEGIES, Paths, run_preflight


LEGAL_MOVEMENT_COUNT = 12
PRIMARY_STRATEGY = "untargeted_selection"
HG_STRATEGY = "hg_expected_aware_selection"


def _verify_persisted_assignments(paths: Paths) -> dict[str, Any]:
    manifest_path = paths.results / "clustering_run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Persisted clustering manifest is required before evaluation.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("reference_labels_read") is not False:
        raise ValueError("Clustering manifest does not prove label isolation.")
    if manifest.get("evaluation_started") is not False:
        raise ValueError("Clustering manifest was unexpectedly mutated.")
    for filename in (
        "cluster_assignments.csv",
        "cluster_assignments.parquet",
        "data_access_log.jsonl",
    ):
        path = paths.results / filename
        expected = manifest["output_files"][filename]
        actual = protocol_io.sha256_file(path)
        if actual != expected:
            raise ValueError(f"Persisted assignment artifact changed: {filename}")
    if manifest["assignment_rows"] != 27_393 * len(METHODS) * len(STRATEGIES):
        raise ValueError("Persisted assignment row count is invalid.")
    return manifest


def _append_access_record(paths: Paths, record: dict[str, Any]) -> None:
    log_path = paths.results / "data_access_log.jsonl"
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    protocol_io.write_text_atomic(
        log_path,
        existing + json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n",
    )


def _purity(reference: pd.Series, clusters: pd.Series) -> float:
    table = pd.crosstab(clusters, reference)
    total = int(table.to_numpy().sum())
    return float(table.max(axis=1).sum() / total) if total else float("nan")


def _mapping_metrics(
    reference: pd.Series,
    clusters: pd.Series,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, list[int], list[str]]:
    classes = sorted(reference.astype(str).unique())
    non_noise_clusters = sorted(int(value) for value in clusters.unique() if int(value) >= 0)
    matrix = np.zeros((len(non_noise_clusters), len(classes)), dtype=int)
    for row_index, cluster in enumerate(non_noise_clusters):
        for column_index, movement in enumerate(classes):
            matrix[row_index, column_index] = int(
                ((clusters == cluster) & (reference == movement)).sum()
            )
    mapping: dict[int, str] = {}
    overlap_by_cluster: dict[int, int] = {}
    if matrix.size:
        rows, columns = linear_sum_assignment(-matrix)
        for row, column in zip(rows, columns, strict=True):
            mapping[non_noise_clusters[row]] = classes[column]
            overlap_by_cluster[non_noise_clusters[row]] = int(matrix[row, column])
    predicted = clusters.map(mapping).fillna("__UNMATCHED_CLUSTER_OR_NOISE__")
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        reference,
        predicted,
        labels=classes,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        reference,
        predicted,
        labels=classes,
        average="weighted",
        zero_division=0,
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        reference,
        predicted,
        labels=classes,
        average=None,
        zero_division=0,
    )
    per_movement = pd.DataFrame(
        {
            "reference_movement_id": classes,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )
    mapping_rows = []
    for cluster in non_noise_clusters:
        movement = mapping.get(cluster, "")
        mapping_rows.append(
            {
                "cluster_label": cluster,
                "mapped_reference_movement_id": movement,
                "matched_overlap": overlap_by_cluster.get(cluster, 0),
                "cluster_valid_reference_support": int((clusters == cluster).sum()),
                "mapping_status": "matched" if movement else "unmatched_cluster",
            }
        )
    matched_movements = set(mapping.values())
    unmatched_clusters = [cluster for cluster in non_noise_clusters if cluster not in mapping]
    unmatched_movements = [movement for movement in classes if movement not in matched_movements]
    summary = {
        "mapped_accuracy": float((predicted == reference).mean()),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
    }
    return (
        summary,
        per_movement,
        pd.DataFrame(mapping_rows),
        unmatched_clusters,
        unmatched_movements,
    )


def evaluate_partition(
    all_assignments: pd.DataFrame,
    valid_reference: pd.DataFrame,
    observed_movement_count: int,
    frozen_hg_target: int,
    cluster_metrics: dict[str, float] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    joined = valid_reference[["scene_id", "trajectory_id", "reference_movement_id"]].merge(
        all_assignments[["scene_id", "trajectory_id", "cluster_label", "is_noise"]],
        on=["scene_id", "trajectory_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if (joined["_merge"] != "both").any():
        raise ValueError("Valid references are missing persisted cluster assignments.")
    reference = joined["reference_movement_id"].astype(str)
    clusters = joined["cluster_label"].astype(int)
    mapping_summary, per_movement, mapping, unmatched_clusters, unmatched_movements = (
        _mapping_metrics(reference, clusters)
    )
    non_noise = all_assignments[all_assignments["cluster_label"] >= 0]
    n_clusters = int(non_noise["cluster_label"].nunique())
    counts = non_noise["cluster_label"].value_counts()
    noise_count = int(all_assignments["is_noise"].sum())
    valid_noise_count = int(joined["is_noise"].sum())
    metrics: dict[str, Any] = {
        "n_non_noise_clusters": n_clusters,
        "observed_reference_movement_count": int(observed_movement_count),
        "legal_movement_count": LEGAL_MOVEMENT_COUNT,
        "frozen_hg_target": int(frozen_hg_target),
        "observed_target_signed_error": n_clusters - int(observed_movement_count),
        "observed_target_abs_error": abs(n_clusters - int(observed_movement_count)),
        "legal_target_signed_error": n_clusters - LEGAL_MOVEMENT_COUNT,
        "legal_target_abs_error": abs(n_clusters - LEGAL_MOVEMENT_COUNT),
        "hg_target_signed_error": n_clusters - int(frozen_hg_target),
        "hg_target_abs_error": abs(n_clusters - int(frozen_hg_target)),
        "noise_count_all_test": noise_count,
        "noise_pct_all_test": 100.0 * noise_count / len(all_assignments),
        "noise_count_valid_reference": valid_noise_count,
        "noise_pct_valid_reference": 100.0 * valid_noise_count / len(joined),
        "largest_cluster_ratio": float(counts.max() / len(non_noise)) if len(non_noise) else np.nan,
        "ari": float(adjusted_rand_score(reference, clusters)),
        "nmi": float(normalized_mutual_info_score(reference, clusters)),
        "purity": _purity(reference, clusters),
        "homogeneity": float(homogeneity_score(reference, clusters)),
        "completeness": float(completeness_score(reference, clusters)),
        "v_measure": float(v_measure_score(reference, clusters)),
        "unmatched_cluster_count": len(unmatched_clusters),
        "unmatched_reference_movement_count": len(unmatched_movements),
        "unmatched_clusters_json": json.dumps(unmatched_clusters),
        "unmatched_reference_movements_json": json.dumps(unmatched_movements),
        **mapping_summary,
    }
    if cluster_metrics:
        metrics.update(cluster_metrics)
        frozen_score_inputs = {
            "hg_estimated_target": int(frozen_hg_target),
            "cluster_count_error": abs(n_clusters - int(frozen_hg_target)),
            "pct_outliers": metrics["noise_pct_all_test"],
            "largest_cluster_ratio": metrics["largest_cluster_ratio"],
            "silhouette_clustered_only": metrics["silhouette_clustered_only"],
            "davies_bouldin_clustered_only": metrics["davies_bouldin_clustered_only"],
        }
        metrics["EMAS_HG"] = core.emas_hg(frozen_score_inputs)
    return metrics, per_movement, mapping


def _test_metadata(paths: Paths) -> pd.DataFrame:
    membership = protocol_io.load_split_membership(paths.evaluation_split, "test", SCENES)
    metadata = protocol_io.load_manifest_metadata(paths.trajectory_manifest, membership)
    protocol_io.assert_phase_rows(metadata, "test")
    return metadata


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def _bootstrap_scene_differences(paired: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260806)
    metrics = (
        "observed_target_abs_error",
        "legal_target_abs_error",
        "ari",
        "nmi",
        "purity",
        "macro_f1",
        "weighted_f1",
        "noise_pct_all_test",
        "silhouette_clustered_only",
        "davies_bouldin_clustered_only",
        "EMAS_HG",
    )
    rows = []
    for method in METHODS:
        method_frame = paired[paired["method"] == method]
        if method_frame["scene_id"].nunique() != 5:
            raise ValueError(f"Scene bootstrap requires five scenes for {method}")
        for metric in metrics:
            values = method_frame[f"delta_{metric}_hg_minus_untargeted"].to_numpy(float)
            samples = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "scene_count": len(values),
                    "mean_paired_delta": float(values.mean()),
                    "median_paired_delta": float(np.median(values)),
                    "scene_bootstrap_mean_ci_lower_2_5": float(np.quantile(samples, 0.025)),
                    "scene_bootstrap_mean_ci_upper_97_5": float(np.quantile(samples, 0.975)),
                    "bootstrap_repetitions": 10_000,
                    "experimental_unit": "scene",
                }
            )
    return pd.DataFrame(rows)


def _paired_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "observed_target_abs_error",
        "legal_target_abs_error",
        "ari",
        "nmi",
        "purity",
        "macro_f1",
        "weighted_f1",
        "noise_pct_all_test",
        "silhouette_clustered_only",
        "davies_bouldin_clustered_only",
        "EMAS_HG",
    ]
    rows = []
    for (scene, method), frame in metrics.groupby(["scene_id", "method"], sort=False):
        indexed = frame.set_index("selection_strategy")
        if set(indexed.index) != set(STRATEGIES):
            raise ValueError(f"Missing paired strategy for {scene}/{method}")
        row: dict[str, Any] = {"scene_id": scene, "method": method}
        for metric in metric_columns:
            untargeted = float(indexed.loc[PRIMARY_STRATEGY, metric])
            aware = float(indexed.loc[HG_STRATEGY, metric])
            row[f"untargeted_{metric}"] = untargeted
            row[f"hg_aware_{metric}"] = aware
            row[f"delta_{metric}_hg_minus_untargeted"] = aware - untargeted
        rows.append(row)
    return pd.DataFrame(rows)


def _target_validation(references: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scene in SCENES:
        valid = references[
            (references["scene_id"] == scene) & (references["reference_status"] == "valid")
        ]
        observed = int(valid["reference_movement_id"].nunique())
        target_values = selected.loc[selected["scene"] == scene, "hg_estimated_target"].unique()
        if len(target_values) != 1:
            raise ValueError(f"Frozen HG target differs within {scene}")
        target = int(target_values[0])
        rows.append(
            {
                "scene_id": scene,
                "frozen_hg_target": target,
                "legal_movement_count": LEGAL_MOVEMENT_COUNT,
                "observed_independent_test_movement_count": observed,
                "hg_target_signed_error_vs_observed": target - observed,
                "hg_target_abs_error_vs_observed": abs(target - observed),
                "hg_target_signed_error_vs_legal": target - LEGAL_MOVEMENT_COUNT,
                "hg_target_abs_error_vs_legal": abs(target - LEGAL_MOVEMENT_COUNT),
            }
        )
    return pd.DataFrame(rows)


def run_primary_evaluation(paths: Paths) -> dict[str, Any]:
    preflight = run_preflight(paths, write_report=False)
    clustering_manifest = _verify_persisted_assignments(paths)
    assignments_path = paths.results / "cluster_assignments.parquet"
    assignments = pd.read_parquet(assignments_path)
    assignment_loaded_at = protocol_io.utc_timestamp()
    if len(assignments) != 27_393 * 6:
        raise ValueError("Assignment table is incomplete.")

    reference_hash = protocol_io.sha256_file(paths.reference_export)
    if reference_hash != preflight["reference_export_sha256"]:
        raise ValueError("Reference export changed after clustering persistence.")
    references = pd.read_csv(paths.reference_export, keep_default_na=False)
    reference_loaded_at = protocol_io.utc_timestamp()
    if set(references["split"]) != {"independent_test"}:
        raise ValueError("Primary evaluation received non-test references.")
    _append_access_record(
        paths,
        {
            "phase": "independent_reference_evaluation",
            "input_kind": "polygon_rule_reference_labels",
            "input_path": paths.reference_export.relative_to(paths.repo_root).as_posix(),
            "checksum_sha256": reference_hash,
            "row_count": len(references),
            "allowed_split": "independent_test",
            "observed_split_values": sorted(references["split"].unique()),
            "assignments_persisted_at_utc": clustering_manifest["assignments_persisted_at_utc"],
            "assignments_loaded_at_utc": assignment_loaded_at,
            "reference_loaded_at_utc": reference_loaded_at,
        },
    )

    selected = pd.read_csv(paths.selected_configurations)
    metadata = _test_metadata(paths)
    metric_rows: list[dict[str, Any]] = []
    movement_tables: list[pd.DataFrame] = []
    mapping_tables: list[pd.DataFrame] = []
    for scene in SCENES:
        scene_assignments = assignments[assignments["scene_id"] == scene]
        scene_reference = references[references["scene_id"] == scene]
        valid_reference = scene_reference[scene_reference["reference_status"] == "valid"]
        observed = int(valid_reference["reference_movement_id"].nunique())
        scene_metadata = metadata[metadata["scene_id"] == scene].copy()
        feature_frame = protocol_io.load_scene_features(
            paths.repo_root, scene_metadata, "test", None
        )
        selected_scene = selected[selected["scene"] == scene]
        normalization = json.loads(selected_scene["normalization_parameters_json"].iloc[0])
        features, _ = core.isotropic_normalize(
            feature_frame[list(core.FEATURE_COLUMNS)].to_numpy(float), normalization
        )
        feature_index = pd.Series(
            np.arange(len(feature_frame)), index=feature_frame["trajectory_id"]
        )
        for method in METHODS:
            for strategy in STRATEGIES:
                run = scene_assignments[
                    (scene_assignments["method"] == method)
                    & (scene_assignments["selection_strategy"] == strategy)
                ].copy()
                run = run.sort_values("trajectory_id", kind="mergesort")
                indices = feature_index.loc[run["trajectory_id"]].to_numpy(int)
                labels = run["cluster_label"].to_numpy(int)
                row = selected_scene[
                    (selected_scene["method"] == method)
                    & (selected_scene["selection_strategy"] == strategy)
                ].iloc[0]
                cluster_metrics = core.safe_cluster_metrics(
                    features[indices],
                    labels,
                    int(row["fit_random_seed"]),
                    3000,
                )
                partition, per_movement, mapping = evaluate_partition(
                    run,
                    valid_reference,
                    observed,
                    int(row["hg_estimated_target"]),
                    cluster_metrics,
                )
                coverage = {
                    "total_test_trajectories": len(scene_reference),
                    "valid_reference_trajectories": len(valid_reference),
                    "excluded_reference_trajectories": len(scene_reference) - len(valid_reference),
                    "reference_coverage_pct": 100.0 * len(valid_reference) / len(scene_reference),
                    "cluster_assignment_count": len(run),
                    "cluster_assignment_coverage_pct": 100.0 * len(run) / len(scene_reference),
                }
                identifier = {
                    "scene_id": scene,
                    "method": method,
                    "selection_strategy": strategy,
                    "frozen_configuration_id": run["frozen_configuration_id"].iloc[0],
                }
                metric_rows.append({**identifier, **coverage, **partition})
                movement_tables.append(per_movement.assign(**identifier))
                mapping_tables.append(mapping.assign(**identifier))

    metrics = pd.DataFrame(metric_rows)
    per_movement = pd.concat(movement_tables, ignore_index=True)
    mapping = pd.concat(mapping_tables, ignore_index=True)
    paired = _paired_differences(metrics)
    bootstrap = _bootstrap_scene_differences(paired)
    targets = _target_validation(references, selected)
    outputs = {
        "independent_test_metrics.csv": metrics,
        "per_movement_metrics.csv": per_movement,
        "cluster_movement_mapping.csv": mapping,
        "paired_strategy_differences.csv": paired,
        "scene_level_bootstrap_ci.csv": bootstrap,
        "target_estimation_validation.csv": targets,
    }
    for filename, frame in outputs.items():
        _write_csv(frame, paths.results / filename)

    evaluation_manifest = {
        "task_name": "future-transportation-independent-test-evaluation-v1",
        "evaluation_timestamp_utc": protocol_io.utc_timestamp(),
        "assignments_persisted_at_utc": clustering_manifest["assignments_persisted_at_utc"],
        "assignments_loaded_before_reference": True,
        "reference_loaded_at_utc": reference_loaded_at,
        "assignment_file_sha256": protocol_io.sha256_file(assignments_path),
        "reference_file_sha256": reference_hash,
        "reference_split": "independent_test",
        "valid_reference_only_for_metrics": True,
        "noise_partition_policy": (
            "Noise label -1 is retained in ARI/NMI/purity/homogeneity/completeness/V-measure; "
            "noise is excluded from Hungarian cluster-to-movement matching and remains unmatched."
        ),
        "hungarian_mapping_policy": (
            "One-to-one maximum-overlap mapping per scene/method/strategy; unmatched clusters "
            "predict an explicit unmatched token and unmatched movements receive zero recall."
        ),
        "emas_hg_role": "development/ranking score, not independent validation",
        "output_checksums": {
            filename: protocol_io.sha256_file(paths.results / filename) for filename in outputs
        },
    }
    protocol_io.write_json_atomic(
        paths.results / "evaluation_run_manifest.json", evaluation_manifest
    )
    return evaluation_manifest


def run_reference_sensitivity(paths: Paths) -> pd.DataFrame:
    _verify_persisted_assignments(paths)
    assignments = pd.read_parquet(paths.results / "cluster_assignments.parquet")
    primary = pd.read_csv(paths.reference_export, keep_default_na=False)
    sensitivity_path = (
        paths.publication_root / "annotations/reference_labels/polygon_assignment_sensitivity.csv"
    )
    sensitivity = pd.read_csv(sensitivity_path, keep_default_na=False)
    sensitivity = sensitivity[sensitivity["split"] == "independent_test"].copy()
    selected = pd.read_csv(paths.selected_configurations)
    rows: list[dict[str, Any]] = []
    variants = ["primary", *sorted(sensitivity["variant"].unique())]
    primary_metrics = pd.read_csv(paths.results / "independent_test_metrics.csv")
    for variant in variants:
        if variant == "primary":
            reference_variant = primary[
                [
                    "scene_id",
                    "trajectory_id",
                    "reference_status",
                    "reference_movement_id",
                ]
            ].copy()
        else:
            selected_variant = sensitivity[sensitivity["variant"] == variant]
            reference_variant = selected_variant.rename(
                columns={
                    "variant_reference_status": "reference_status",
                    "variant_movement_id": "reference_movement_id",
                }
            )[["scene_id", "trajectory_id", "reference_status", "reference_movement_id"]]
        for scene in SCENES:
            scene_reference = reference_variant[reference_variant["scene_id"] == scene]
            valid = scene_reference[scene_reference["reference_status"] == "valid"]
            observed = int(valid["reference_movement_id"].nunique())
            target = int(selected.loc[selected["scene"] == scene, "hg_estimated_target"].iloc[0])
            for method in METHODS:
                for strategy in STRATEGIES:
                    run = assignments[
                        (assignments["scene_id"] == scene)
                        & (assignments["method"] == method)
                        & (assignments["selection_strategy"] == strategy)
                    ]
                    partition, _, _ = evaluate_partition(run, valid, observed, target)
                    baseline = primary_metrics[
                        (primary_metrics["scene_id"] == scene)
                        & (primary_metrics["method"] == method)
                        & (primary_metrics["selection_strategy"] == strategy)
                    ].iloc[0]
                    rows.append(
                        {
                            "reference_variant": variant,
                            "scene_id": scene,
                            "method": method,
                            "selection_strategy": strategy,
                            "total_test_trajectories": len(scene_reference),
                            "valid_reference_trajectories": len(valid),
                            "valid_reference_coverage_pct": 100.0
                            * len(valid)
                            / len(scene_reference),
                            "observed_reference_movement_count": observed,
                            "ari": partition["ari"],
                            "nmi": partition["nmi"],
                            "purity": partition["purity"],
                            "macro_f1": partition["macro_f1"],
                            "delta_ari_vs_primary": partition["ari"] - float(baseline["ari"]),
                            "delta_nmi_vs_primary": partition["nmi"] - float(baseline["nmi"]),
                            "delta_purity_vs_primary": partition["purity"]
                            - float(baseline["purity"]),
                            "delta_macro_f1_vs_primary": partition["macro_f1"]
                            - float(baseline["macro_f1"]),
                        }
                    )
    result = pd.DataFrame(rows)
    for metric in ("ari", "nmi", "purity", "macro_f1"):
        lookup = result.pivot_table(
            index=["reference_variant", "scene_id", "method"],
            columns="selection_strategy",
            values=metric,
        )
        delta = (lookup[HG_STRATEGY] - lookup[PRIMARY_STRATEGY]).rename(
            f"hg_minus_untargeted_{metric}"
        )
        result = result.merge(
            delta.reset_index(),
            on=["reference_variant", "scene_id", "method"],
            how="left",
            validate="many_to_one",
        )
    _write_csv(result, paths.results / "reference_sensitivity_evaluation.csv")
    return result


def create_figures(paths: Paths) -> None:
    metrics = pd.read_csv(paths.results / "independent_test_metrics.csv")
    targets = pd.read_csv(paths.results / "target_estimation_validation.csv")
    paths.figures.mkdir(parents=True, exist_ok=True)
    scene_labels = {
        "bellevue_116th_ne12th": "116th",
        "bellevue_150th_newport": "Newport",
        "bellevue_150th_eastgate": "Eastgate",
        "bellevue_150th_se38th": "SE38th",
        "bellevue_ne8th": "NE8th",
    }
    colors = {PRIMARY_STRATEGY: "#4c78a8", HG_STRATEGY: "#e45756"}

    def save(fig: Any, stem: str) -> None:
        fig.savefig(paths.figures / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(paths.figures / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, method in zip(axes, METHODS, strict=True):
        subset = metrics[metrics["method"] == method]
        x = np.arange(len(SCENES))
        for offset, strategy in ((-0.18, PRIMARY_STRATEGY), (0.18, HG_STRATEGY)):
            values = (
                subset[subset["selection_strategy"] == strategy]
                .set_index("scene_id")
                .loc[list(SCENES), "observed_target_abs_error"]
            )
            ax.bar(
                x + offset,
                values,
                width=0.36,
                color=colors[strategy],
                label=strategy.replace("_selection", ""),
            )
        ax.set_title(method.upper())
        ax.set_xticks(x, [scene_labels[s] for s in SCENES], rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Absolute error vs observed movements")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Independent-test cluster-count alignment")
    save(fig, "01_observed_target_error_by_scene")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ax, metric in zip(axes.flat, ("ari", "nmi", "purity", "macro_f1"), strict=True):
        x = np.arange(len(SCENES))
        for method_index, method in enumerate(METHODS):
            for strategy_index, strategy in enumerate(STRATEGIES):
                subset = (
                    metrics[
                        (metrics["method"] == method) & (metrics["selection_strategy"] == strategy)
                    ]
                    .set_index("scene_id")
                    .loc[list(SCENES)]
                )
                offset = (method_index * 2 + strategy_index - 2.5) * 0.11
                ax.plot(
                    x + offset,
                    subset[metric],
                    marker="o" if strategy_index == 0 else "s",
                    linestyle="none",
                    color=colors[strategy],
                    alpha=0.85,
                    label=f"{method}/{strategy.split('_')[0]}",
                )
        ax.set_title(metric.upper().replace("MACRO_F1", "Macro F1"))
        ax.set_xticks(x, [scene_labels[s] for s in SCENES], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8)
    fig.suptitle("Independent polygon-reference agreement by scene")
    fig.subplots_adjust(bottom=0.16)
    save(fig, "02_independent_agreement_metrics")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, method in zip(axes, METHODS, strict=True):
        subset = metrics[metrics["method"] == method]
        x = np.arange(len(SCENES))
        for offset, strategy in ((-0.18, PRIMARY_STRATEGY), (0.18, HG_STRATEGY)):
            values = (
                subset[subset["selection_strategy"] == strategy]
                .set_index("scene_id")
                .loc[list(SCENES), "noise_pct_all_test"]
            )
            ax.bar(x + offset, values, width=0.36, color=colors[strategy])
        ax.set_title(method.upper())
        ax.set_xticks(x, [scene_labels[s] for s in SCENES], rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Noise trajectories (%)")
    fig.suptitle("Independent-test outlier percentage")
    save(fig, "03_outlier_percentage_by_scene_method")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SCENES))
    ax.plot(
        x,
        targets.set_index("scene_id").loc[list(SCENES), "frozen_hg_target"],
        "o-",
        label="Frozen HG target",
        color="#e45756",
    )
    ax.plot(
        x,
        targets.set_index("scene_id").loc[list(SCENES), "observed_independent_test_movement_count"],
        "s-",
        label="Observed reference",
        color="#4c78a8",
    )
    ax.plot(
        x,
        targets.set_index("scene_id").loc[list(SCENES), "legal_movement_count"],
        "^-",
        label="Legal mapping count",
        color="#59a14f",
    )
    ax.set_xticks(x, [scene_labels[s] for s in SCENES])
    ax.set_ylabel("Movement count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.set_title("Frozen target versus independent observed and legal counts")
    save(fig, "04_target_estimation_comparison")

    fig, ax = plt.subplots(figsize=(9, 6))
    markers = {"kmeans": "o", "hdbscan": "s", "optics": "^"}
    for method in METHODS:
        for strategy in STRATEGIES:
            subset = metrics[
                (metrics["method"] == method) & (metrics["selection_strategy"] == strategy)
            ]
            ax.scatter(
                subset["observed_target_abs_error"],
                subset["macro_f1"],
                marker=markers[method],
                color=colors[strategy],
                s=65,
                alpha=0.8,
                label=f"{method}/{strategy.split('_')[0]}",
            )
    ax.set_xlabel("Absolute cluster-count error vs observed movements")
    ax.set_ylabel("Mapped macro F1")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    ax.legend(unique.values(), unique.keys(), fontsize=8, ncol=2)
    ax.set_title("Target alignment and independent agreement trade-off")
    save(fig, "05_target_alignment_vs_macro_f1")
