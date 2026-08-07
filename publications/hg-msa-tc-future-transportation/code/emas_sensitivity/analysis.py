"""Reproduce and stress-test EMAS_HG-v1 without changing frozen selections."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau, spearmanr

from metrics.emas_hg import (
    EMAS_HG_VERSION,
    ORIGINAL_WEIGHTS,
    EmasWeights,
    cluster_balance_component,
    compute_emas_hg,
    davies_bouldin_component,
    non_outlier_component,
    silhouette_component,
    target_agreement_component,
)


COMPONENTS = ("T", "O", "B", "S", "D")
REPRODUCTION_TOLERANCE = 1e-12
IMMUTABLE_INPUTS = (
    "configs/frozen_evaluation_protocol.yaml",
    "configs/split_aware_runner.yaml",
    "results/development/model_selection_candidates.csv",
    "results/development/selected_configurations.csv",
    "results/development/frozen_selection_manifest.json",
    "results/independent_test/cluster_assignments.csv",
    "results/independent_test/cluster_assignments.parquet",
    "results/independent_test/independent_test_metrics.csv",
    "annotations/reference_labels/independent_test_reference_labels.csv",
)


@dataclass(frozen=True)
class Paths:
    publication: Path
    repo: Path
    config: Path
    results: Path
    figures: Path
    docs: Path


def default_paths() -> Paths:
    publication = Path(__file__).resolve().parents[2]
    return Paths(
        publication=publication,
        repo=publication.parents[1],
        config=publication / "configs" / "emas_weight_scenarios.yaml",
        results=publication / "results" / "emas",
        figures=publication / "figures" / "emas",
        docs=publication / "docs",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def git_head(repo: Path) -> str:
    command = [
        "git",
        "-c",
        f"safe.directory={repo.as_posix()}",
        "rev-parse",
        "HEAD",
    ]
    return subprocess.check_output(command, cwd=repo, text=True).strip()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def create_input_manifest(paths: Paths) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for relative in IMMUTABLE_INPUTS:
        path = paths.publication / relative
        if not path.exists():
            raise FileNotFoundError(f"Required immutable Task 06 input is missing: {path}")
        files[relative] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    payload = {
        "task": "futuretransp-emas-formalization-sensitivity",
        "created_at_utc": utc_timestamp(),
        "git_commit_before_analysis": git_head(paths.repo),
        "score_version": EMAS_HG_VERSION,
        "semantic_reference_labels_read": False,
        "independent_test_clustering_executed": False,
        "files": files,
    }
    write_json(paths.results / "task_06_input_manifest.json", payload)
    return payload


def verify_immutable_inputs(paths: Paths, manifest: dict[str, Any]) -> None:
    mismatches: list[str] = []
    for relative, metadata in manifest["files"].items():
        current = sha256_file(paths.publication / relative)
        if current != metadata["sha256"]:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(f"Frozen Task 06 inputs changed: {mismatches}")


def _normalized_inputs(row: pd.Series | dict[str, Any], source: str) -> dict[str, Any]:
    if source == "independent_test_metrics":
        return {
            "expected_target": row["frozen_hg_target"],
            "cluster_count_error": row["hg_target_abs_error"],
            "pct_outliers": row["noise_pct_all_test"],
            "largest_cluster_ratio": row.get("largest_cluster_ratio"),
            "silhouette_clustered_only": row.get("silhouette_clustered_only"),
            "davies_bouldin_clustered_only": row.get("davies_bouldin_clustered_only"),
        }
    return {
        "expected_target": row["hg_estimated_target"],
        "cluster_count_error": row["cluster_count_error"],
        "pct_outliers": row["pct_outliers"],
        "largest_cluster_ratio": row.get("largest_cluster_ratio"),
        "silhouette_clustered_only": row.get("silhouette_clustered_only"),
        "davies_bouldin_clustered_only": row.get("davies_bouldin_clustered_only"),
    }


def _reproduction_rows(
    frame: pd.DataFrame, source: str, strategy_column: str
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source_row, row in frame.iterrows():
        inputs = _normalized_inputs(row, source)
        recomputed = compute_emas_hg(**inputs)
        stored = float(row["EMAS_HG"])
        difference = abs(recomputed - stored)
        components = {
            "T": target_agreement_component(
                inputs["expected_target"], inputs["cluster_count_error"]
            ),
            "O": non_outlier_component(inputs["pct_outliers"]),
            "B": cluster_balance_component(inputs["largest_cluster_ratio"]),
            "S": silhouette_component(inputs["silhouette_clustered_only"]),
            "D": davies_bouldin_component(inputs["davies_bouldin_clustered_only"]),
        }
        output.append(
            {
                "source": source,
                "source_row": int(source_row),
                "scene": row.get("scene", row.get("scene_id", "")),
                "method": row.get("method", ""),
                "selection_strategy": row.get(strategy_column, ""),
                "trial_index": row.get("trial_index", np.nan),
                **inputs,
                **components,
                "stored_EMAS_HG": stored,
                "recomputed_EMAS_HG": recomputed,
                "absolute_difference": difference,
                "tolerance": REPRODUCTION_TOLERANCE,
                "within_tolerance": bool(difference <= REPRODUCTION_TOLERANCE),
            }
        )
    return output


def run_reproduction_check(paths: Paths) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidates = pd.read_csv(
        paths.publication / "results/development/model_selection_candidates.csv"
    )
    rows.extend(_reproduction_rows(candidates, "development_candidates", ""))
    selected = pd.read_csv(paths.publication / "results/development/selected_configurations.csv")
    rows.extend(_reproduction_rows(selected, "development_selected", "selection_strategy"))
    frozen = yaml.safe_load(
        (paths.publication / "configs/frozen_evaluation_protocol.yaml").read_text(encoding="utf-8")
    )
    rows.extend(
        _reproduction_rows(
            pd.DataFrame(frozen["selected_configurations"]),
            "frozen_protocol_selected",
            "selection_strategy",
        )
    )
    independent = pd.read_csv(
        paths.publication / "results/independent_test/independent_test_metrics.csv"
    )
    rows.extend(_reproduction_rows(independent, "independent_test_metrics", "selection_strategy"))
    output = pd.DataFrame(rows)
    write_csv(output, paths.results / "emas_reproduction_check.csv")
    maximum = float(output["absolute_difference"].max())
    if maximum > REPRODUCTION_TOLERANCE:
        raise RuntimeError(
            f"EMAS reproduction failed: max difference {maximum:.17g} exceeds "
            f"{REPRODUCTION_TOLERANCE:.1e}."
        )
    return output


def load_named_weights(paths: Paths) -> dict[str, EmasWeights]:
    config = yaml.safe_load(paths.config.read_text(encoding="utf-8"))
    scenarios = {
        name: EmasWeights.from_mapping(values) for name, values in config["named_scenarios"].items()
    }
    if scenarios["original"] != ORIGINAL_WEIGHTS:
        raise ValueError("The named original scenario does not match EMAS_HG-v1.")
    return scenarios


def create_weight_samples(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = yaml.safe_load(paths.config.read_text(encoding="utf-8"))
    local = config["local_grid"]
    step = float(local["step"])
    units = round(1.0 / step)
    bounds = {
        name: (round(values[0] / step), round(values[1] / step))
        for name, values in local.items()
        if name in COMPONENTS
    }
    local_rows: list[dict[str, Any]] = []
    for t in range(bounds["T"][0], bounds["T"][1] + 1):
        for o in range(bounds["O"][0], bounds["O"][1] + 1):
            for b in range(bounds["B"][0], bounds["B"][1] + 1):
                for s in range(bounds["S"][0], bounds["S"][1] + 1):
                    d = units - t - o - b - s
                    if bounds["D"][0] <= d <= bounds["D"][1]:
                        values = [value * step for value in (t, o, b, s, d)]
                        weights = EmasWeights(*values)
                        weights.validate()
                        local_rows.append(weights.as_dict())
    local_frame = pd.DataFrame(local_rows).drop_duplicates().sort_values(list(COMPONENTS))
    local_frame.insert(0, "weight_id", [f"local_{index:04d}" for index in range(len(local_frame))])
    local_frame.insert(1, "family", "local")
    write_csv(local_frame, paths.results / "emas_local_weight_grid.csv")

    global_config = config["global_sample"]
    count = int(global_config["unique_vectors"])
    rng = np.random.default_rng(int(global_config["seed"]))
    alpha = np.asarray(global_config["alpha"], dtype=float)
    unique: dict[tuple[float, ...], np.ndarray] = {}
    while len(unique) < count:
        for vector in rng.dirichlet(alpha, size=count - len(unique)):
            unique.setdefault(tuple(np.round(vector, 15)), vector)
    global_rows = [dict(zip(COMPONENTS, vector, strict=True)) for vector in unique.values()]
    global_frame = pd.DataFrame(global_rows).iloc[:count].copy()
    global_frame.insert(
        0, "weight_id", [f"global_{index:04d}" for index in range(len(global_frame))]
    )
    global_frame.insert(1, "family", "global")
    if (global_frame[list(COMPONENTS)] < 0.0).any().any() or not np.allclose(
        global_frame[list(COMPONENTS)].sum(axis=1), 1.0, atol=1e-12
    ):
        raise ValueError("Generated global weights must be non-negative and sum to one.")
    write_csv(global_frame, paths.results / "emas_global_weight_sample.csv")
    return local_frame, global_frame


def load_candidate_components(paths: Paths) -> pd.DataFrame:
    frame = pd.read_csv(paths.publication / "results/development/model_selection_candidates.csv")
    frame["candidate_id"] = frame.apply(
        lambda row: f"{row['scene']}|{row['method']}|trial_{int(row['trial_index']):03d}", axis=1
    )
    inputs = [_normalized_inputs(row, "development_candidates") for _, row in frame.iterrows()]
    frame["T"] = [
        target_agreement_component(item["expected_target"], item["cluster_count_error"])
        for item in inputs
    ]
    frame["O"] = [non_outlier_component(item["pct_outliers"]) for item in inputs]
    frame["B"] = [cluster_balance_component(item["largest_cluster_ratio"]) for item in inputs]
    frame["S"] = [silhouette_component(item["silhouette_clustered_only"]) for item in inputs]
    frame["D"] = [
        davies_bouldin_component(item["davies_bouldin_clustered_only"]) for item in inputs
    ]
    return frame


def _rank(scores: np.ndarray, params: Iterable[str], trials: Iterable[int]) -> np.ndarray:
    order = sorted(
        range(len(scores)),
        key=lambda index: (
            -float(scores[index]),
            str(list(params)[index]),
            int(list(trials)[index]),
        ),
    )
    ranks = np.empty(len(order), dtype=int)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return ranks


def _rank_group(group: pd.DataFrame, weights: EmasWeights) -> tuple[np.ndarray, np.ndarray]:
    scores = group[list(COMPONENTS)].to_numpy(dtype=float) @ np.asarray(weights.as_tuple())
    ranks = _rank(scores, group["params_json"].tolist(), group["trial_index"].tolist())
    return scores, ranks


def run_named_scenarios(paths: Paths, candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scenarios = load_named_weights(paths)
    for (scene, method), group in candidates.groupby(["scene", "method"], sort=True):
        group = group.reset_index(drop=True)
        for name, weights in scenarios.items():
            scores, ranks = _rank_group(group, weights)
            for index, row in group.iterrows():
                rows.append(
                    {
                        "scenario": name,
                        **weights.as_dict(),
                        "scene": scene,
                        "method": method,
                        "candidate_id": row["candidate_id"],
                        "trial_index": int(row["trial_index"]),
                        "params_json": row["params_json"],
                        "score": float(scores[index]),
                        "rank": int(ranks[index]),
                        "is_top_ranked": bool(ranks[index] == 1),
                        "selected_untargeted": bool(row["selected_untargeted"]),
                        "selected_hg_expected_aware": bool(row["selected_hg_expected_aware"]),
                    }
                )
    output = pd.DataFrame(rows)
    write_csv(output, paths.results / "emas_named_scenario_results.csv")
    return output


def _safe_correlation(function: Any, first: np.ndarray, second: np.ndarray) -> float:
    result = function(first, second)
    value = result.statistic if hasattr(result, "statistic") else result[0]
    return float(value) if np.isfinite(value) else 1.0


def run_grid_analysis(
    paths: Paths,
    candidates: pd.DataFrame,
    local_weights: pd.DataFrame,
    global_weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grid_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    all_weights = pd.concat([local_weights, global_weights], ignore_index=True)
    for (scene, method), group in candidates.groupby(["scene", "method"], sort=True):
        group = group.reset_index(drop=True)
        original_scores, original_ranks = _rank_group(group, ORIGINAL_WEIGHTS)
        original_top = str(group.loc[int(np.argmin(original_ranks)), "candidate_id"])
        original_top3 = set(group.loc[original_ranks <= 3, "candidate_id"])
        score_matrix = (
            group[list(COMPONENTS)].to_numpy(dtype=float)
            @ all_weights[list(COMPONENTS)].to_numpy(dtype=float).T
        )
        top_by_vector: list[str] = []
        correlations: list[dict[str, Any]] = []
        for vector_index, weight_row in all_weights.iterrows():
            scores = score_matrix[:, vector_index]
            ranks = _rank(scores, group["params_json"].tolist(), group["trial_index"].tolist())
            top_index = int(np.argmin(ranks))
            top_candidate = str(group.loc[top_index, "candidate_id"])
            top_by_vector.append(top_candidate)
            top3 = set(group.loc[ranks <= 3, "candidate_id"])
            row = {
                "family": weight_row["family"],
                "weight_id": weight_row["weight_id"],
                **{name: float(weight_row[name]) for name in COMPONENTS},
                "scene": scene,
                "method": method,
                "top_candidate_id": top_candidate,
                "top_trial_index": int(group.loc[top_index, "trial_index"]),
                "top_params_json": group.loc[top_index, "params_json"],
                "original_top_candidate_id": original_top,
                "original_top_preserved": bool(top_candidate == original_top),
                "spearman_vs_original": _safe_correlation(spearmanr, original_scores, scores),
                "kendall_vs_original": _safe_correlation(kendalltau, original_scores, scores),
                "top3_overlap_fraction": len(original_top3 & top3) / 3.0,
            }
            grid_rows.append(row)
            correlations.append(row)
        for candidate_index, candidate in group.iterrows():
            values = score_matrix[candidate_index, :]
            sensitivity_rows.append(
                {
                    "scene": scene,
                    "method": method,
                    "candidate_id": candidate["candidate_id"],
                    "trial_index": int(candidate["trial_index"]),
                    "params_json": candidate["params_json"],
                    "original_score": float(original_scores[candidate_index]),
                    "original_rank": int(original_ranks[candidate_index]),
                    "minimum_score": float(values.min()),
                    "maximum_score": float(values.max()),
                    "mean_score": float(values.mean()),
                    "std_score": float(values.std(ddof=0)),
                    "score_range": float(values.max() - values.min()),
                }
            )
        ordered = np.argsort(original_ranks)
        first_index, second_index = int(ordered[0]), int(ordered[1])
        group_grid = pd.DataFrame(correlations)
        selected_untargeted_index = int(group.index[group["selected_untargeted"]][0])
        selected_hg_index = int(group.index[group["selected_hg_expected_aware"]][0])
        margin_rows.append(
            {
                "scene": scene,
                "method": method,
                "original_top_candidate_id": original_top,
                "original_second_candidate_id": group.loc[second_index, "candidate_id"],
                "original_first_score": float(original_scores[first_index]),
                "original_second_score": float(original_scores[second_index]),
                "original_margin": float(
                    original_scores[first_index] - original_scores[second_index]
                ),
                "frozen_untargeted_candidate_id": group.loc[
                    selected_untargeted_index, "candidate_id"
                ],
                "frozen_untargeted_original_emas_rank": int(
                    original_ranks[selected_untargeted_index]
                ),
                "frozen_hg_aware_candidate_id": group.loc[selected_hg_index, "candidate_id"],
                "frozen_hg_aware_original_emas_rank": int(original_ranks[selected_hg_index]),
                "named_rank_reversal_frequency": np.nan,
                "local_rank_reversal_frequency": float(
                    1.0
                    - group_grid.loc[
                        group_grid["family"] == "local", "original_top_preserved"
                    ].mean()
                ),
                "global_rank_reversal_frequency": float(
                    1.0
                    - group_grid.loc[
                        group_grid["family"] == "global", "original_top_preserved"
                    ].mean()
                ),
                "local_reversal_weight_ids_json": json.dumps(
                    group_grid.loc[
                        (group_grid["family"] == "local") & (~group_grid["original_top_preserved"]),
                        "weight_id",
                    ].tolist()
                ),
                "global_reversal_weight_ids_json": json.dumps(
                    group_grid.loc[
                        (group_grid["family"] == "global")
                        & (~group_grid["original_top_preserved"]),
                        "weight_id",
                    ].tolist()
                ),
            }
        )
    grid_output = pd.DataFrame(grid_rows)
    sensitivity_output = pd.DataFrame(sensitivity_rows)
    margin_output = pd.DataFrame(margin_rows)
    write_csv(grid_output, paths.results / "emas_weight_grid_results.csv")
    write_csv(sensitivity_output, paths.results / "emas_candidate_score_sensitivity.csv")
    write_csv(margin_output, paths.results / "emas_margin_analysis.csv")
    return grid_output, sensitivity_output, margin_output


def create_rank_stability(paths: Paths, grid: pd.DataFrame, named: pd.DataFrame) -> pd.DataFrame:
    named_tops = named[named["is_top_ranked"]].copy()
    original_map = named_tops[named_tops["scenario"] == "original"].set_index(["scene", "method"])[
        "candidate_id"
    ]
    named_tops["original_top_candidate_id"] = [
        original_map.loc[(row.scene, row.method)] for row in named_tops.itertuples()
    ]
    named_tops["original_top_preserved"] = (
        named_tops["candidate_id"] == named_tops["original_top_candidate_id"]
    )
    records: list[dict[str, Any]] = []
    for family, frame, vector_column, top_column in (
        ("named", named_tops, "scenario", "candidate_id"),
        ("local", grid[grid["family"] == "local"], "weight_id", "top_candidate_id"),
        ("global", grid[grid["family"] == "global"], "weight_id", "top_candidate_id"),
    ):
        for (scene, method), group in frame.groupby(["scene", "method"], sort=True):
            counts = group[top_column].value_counts()
            original_top = str(group["original_top_candidate_id"].iloc[0])
            records.append(
                {
                    "family": family,
                    "scene": scene,
                    "method": method,
                    "n_weight_vectors": int(group[vector_column].nunique()),
                    "original_top_candidate_id": original_top,
                    "original_top_frequency": int(counts.get(original_top, 0)),
                    "original_top_stability_pct": float(
                        100.0 * group["original_top_preserved"].mean()
                    ),
                    "n_distinct_top_candidates": int(counts.size),
                    "most_frequent_top_candidate_id": str(counts.index[0]),
                    "most_frequent_top_frequency": int(counts.iloc[0]),
                }
            )
    output = pd.DataFrame(records)
    write_csv(output, paths.results / "emas_rank_stability.csv")
    return output


def create_component_analysis(paths: Paths, candidates: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    scopes = [("all", "ALL", "ALL", candidates)] + [
        ("scene_method", scene, method, group)
        for (scene, method), group in candidates.groupby(["scene", "method"], sort=True)
    ]
    for scope, scene, method, frame in scopes:
        original = frame[list(COMPONENTS)].to_numpy(dtype=float) @ np.asarray(
            ORIGINAL_WEIGHTS.as_tuple()
        )
        for component in COMPONENTS:
            values = frame[component].to_numpy(dtype=float)
            records.extend(
                [
                    {
                        "scope": scope,
                        "scene": scene,
                        "method": method,
                        "statistic": "variance",
                        "variable_x": component,
                        "variable_y": "",
                        "value": float(np.var(values)),
                    },
                    {
                        "scope": scope,
                        "scene": scene,
                        "method": method,
                        "statistic": "pearson_with_original_EMAS",
                        "variable_x": component,
                        "variable_y": "EMAS_HG",
                        "value": float(np.corrcoef(values, original)[0, 1])
                        if np.std(values) > 0
                        else np.nan,
                    },
                    {
                        "scope": scope,
                        "scene": scene,
                        "method": method,
                        "statistic": "fraction_at_zero",
                        "variable_x": component,
                        "variable_y": "",
                        "value": float(np.mean(np.isclose(values, 0.0))),
                    },
                    {
                        "scope": scope,
                        "scene": scene,
                        "method": method,
                        "statistic": "fraction_at_one",
                        "variable_x": component,
                        "variable_y": "",
                        "value": float(np.mean(np.isclose(values, 1.0))),
                    },
                ]
            )
        for first_index, first in enumerate(COMPONENTS):
            for second in COMPONENTS[first_index:]:
                first_values = frame[first].to_numpy(dtype=float)
                second_values = frame[second].to_numpy(dtype=float)
                correlation = (
                    float(np.corrcoef(first_values, second_values)[0, 1])
                    if np.std(first_values) > 0 and np.std(second_values) > 0
                    else (1.0 if first == second else np.nan)
                )
                records.append(
                    {
                        "scope": scope,
                        "scene": scene,
                        "method": method,
                        "statistic": "component_pearson",
                        "variable_x": first,
                        "variable_y": second,
                        "value": correlation,
                    }
                )
    output = pd.DataFrame(records)
    write_csv(output, paths.results / "emas_component_analysis.csv")
    return output


def finalize_margin_named_results(
    paths: Paths, margins: pd.DataFrame, named: pd.DataFrame
) -> pd.DataFrame:
    tops = named[named["is_top_ranked"]].copy()
    original = tops[tops["scenario"] == "original"].set_index(["scene", "method"])["candidate_id"]
    frequencies: dict[tuple[str, str], float] = {}
    reversal_scenarios: dict[tuple[str, str], list[str]] = {}
    for key, group in tops.groupby(["scene", "method"]):
        frequencies[key] = float(1.0 - np.mean(group["candidate_id"] == original.loc[key]))
        reversal_scenarios[key] = group.loc[
            group["candidate_id"] != original.loc[key], "scenario"
        ].tolist()
    margins["named_rank_reversal_frequency"] = [
        frequencies[(row.scene, row.method)] for row in margins.itertuples()
    ]
    margins["named_reversal_scenarios_json"] = [
        json.dumps(reversal_scenarios[(row.scene, row.method)]) for row in margins.itertuples()
    ]
    write_csv(margins, paths.results / "emas_margin_analysis.csv")
    return margins


def run_all(paths: Paths | None = None) -> dict[str, Any]:
    paths = paths or default_paths()
    if paths.results.exists() and any(paths.results.iterdir()):
        raise FileExistsError(f"Task 06 result directory is single-use: {paths.results}")
    paths.results.mkdir(parents=True, exist_ok=True)
    manifest = create_input_manifest(paths)
    reproduction = run_reproduction_check(paths)
    candidates = load_candidate_components(paths)
    local_weights, global_weights = create_weight_samples(paths)
    named = run_named_scenarios(paths, candidates)
    grid, candidate_sensitivity, margins = run_grid_analysis(
        paths, candidates, local_weights, global_weights
    )
    margins = finalize_margin_named_results(paths, margins, named)
    stability = create_rank_stability(paths, grid, named)
    components = create_component_analysis(paths, candidates)
    verify_immutable_inputs(paths, manifest)
    summary = {
        "score_version": EMAS_HG_VERSION,
        "reproduction_rows": int(len(reproduction)),
        "reproduction_max_absolute_error": float(reproduction["absolute_difference"].max()),
        "development_candidates": int(len(candidates)),
        "scene_method_groups": int(candidates.groupby(["scene", "method"]).ngroups),
        "named_weight_scenarios": int(named["scenario"].nunique()),
        "local_weight_vectors": int(len(local_weights)),
        "global_weight_vectors": int(len(global_weights)),
        "grid_group_rows": int(len(grid)),
        "candidate_sensitivity_rows": int(len(candidate_sensitivity)),
        "rank_stability_rows": int(len(stability)),
        "component_analysis_rows": int(len(components)),
        "margin_rows": int(len(margins)),
        "independent_test_clustering_rerun": False,
        "independent_test_reference_labels_read_for_weight_analysis": False,
        "frozen_inputs_unchanged_after_analysis": True,
        "created_at_utc": utc_timestamp(),
    }
    write_json(paths.results / "emas_analysis_manifest.json", summary)
    return summary
