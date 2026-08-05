"""Evaluate frozen clusters against valid polygon-rule reference labels.

This adapter never fits a clustering model. It accepts already frozen assignments and
is intentionally separate from the independent-test clustering runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


REQUIRED_CLUSTER_COLUMNS = {"scene_id", "trajectory_id", "cluster_id"}
REQUIRED_REFERENCE_COLUMNS = {
    "scene_id",
    "trajectory_id",
    "split",
    "reference_status",
    "reference_movement_id",
}


def purity_score(reference: pd.Series, clusters: pd.Series) -> float:
    table = pd.crosstab(clusters, reference)
    return float(table.max(axis=1).sum() / table.to_numpy().sum()) if len(table) else float("nan")


def optimal_cluster_mapping(reference: pd.Series, clusters: pd.Series) -> dict[Any, str]:
    table = pd.crosstab(clusters, reference)
    if table.empty:
        return {}
    rows, columns = linear_sum_assignment(-table.to_numpy())
    return {
        table.index[row]: str(table.columns[column])
        for row, column in zip(rows, columns, strict=True)
    }


def evaluate_assignments(
    cluster_assignments: pd.DataFrame,
    reference_labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return scene metrics, per-movement metrics, and coverage diagnostics."""
    missing_clusters = REQUIRED_CLUSTER_COLUMNS - set(cluster_assignments)
    missing_reference = REQUIRED_REFERENCE_COLUMNS - set(reference_labels)
    if missing_clusters or missing_reference:
        raise ValueError(
            f"Missing columns: clusters={sorted(missing_clusters)}, "
            f"reference={sorted(missing_reference)}"
        )
    if cluster_assignments.duplicated(["scene_id", "trajectory_id"]).any():
        raise ValueError("Cluster assignments are not unique by scene and trajectory.")

    reference = reference_labels[reference_labels["split"] == "independent_test"].copy()
    merged = reference.merge(
        cluster_assignments,
        on=["scene_id", "trajectory_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    scene_rows: list[dict[str, Any]] = []
    movement_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for scene, scene_frame in merged.groupby("scene_id", sort=True):
        valid_reference = scene_frame[scene_frame["reference_status"] == "valid"]
        evaluated = valid_reference[valid_reference["_merge"] == "both"].copy()
        coverage_rows.append(
            {
                "scene_id": scene,
                "independent_test_rows": len(scene_frame),
                "valid_reference_rows": len(valid_reference),
                "evaluated_rows": len(evaluated),
                "excluded_reference_rows": int((scene_frame["reference_status"] != "valid").sum()),
                "missing_cluster_rows": int((valid_reference["_merge"] != "both").sum()),
                "valid_reference_coverage_pct": 100.0 * len(valid_reference) / len(scene_frame),
            }
        )
        if evaluated.empty:
            continue
        reference_values = evaluated["reference_movement_id"].astype(str)
        cluster_values = evaluated["cluster_id"]
        mapping = optimal_cluster_mapping(reference_values, cluster_values)
        predicted = cluster_values.map(mapping).fillna("__unmapped_cluster__")
        scene_rows.append(
            {
                "scene_id": scene,
                "evaluated_rows": len(evaluated),
                "ari": adjusted_rand_score(reference_values, cluster_values),
                "nmi": normalized_mutual_info_score(reference_values, cluster_values),
                "purity": purity_score(reference_values, cluster_values),
                "mapped_accuracy": float((predicted == reference_values).mean()),
                "cluster_count": int(cluster_values.nunique()),
                "observed_reference_movement_count": int(reference_values.nunique()),
            }
        )
        for movement in sorted(reference_values.unique()):
            true_positive = int(((reference_values == movement) & (predicted == movement)).sum())
            false_positive = int(((reference_values != movement) & (predicted == movement)).sum())
            false_negative = int(((reference_values == movement) & (predicted != movement)).sum())
            precision = (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else 0.0
            )
            recall = (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative
                else 0.0
            )
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            movement_rows.append(
                {
                    "scene_id": scene,
                    "reference_movement_id": movement,
                    "support": int((reference_values == movement).sum()),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
    return pd.DataFrame(scene_rows), pd.DataFrame(movement_rows), pd.DataFrame(coverage_rows)


def evaluate_files(
    frozen_cluster_assignments: Path,
    independent_test_reference_labels: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load two existing files and evaluate; no clustering command is available here."""
    return evaluate_assignments(
        pd.read_csv(frozen_cluster_assignments),
        pd.read_csv(independent_test_reference_labels),
    )
