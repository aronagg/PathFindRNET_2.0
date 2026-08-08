"""Development-only orchestration for the frozen HG-SMG-TC protocol."""

from __future__ import annotations

import subprocess
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from target_estimation.hg_target_estimator import estimate_hg_target_detailed

from . import ablations
from .io import (
    ABLATION_PATH,
    PROTOCOL_PATH,
    PUBLICATION_ROOT,
    REPOSITORY_ROOT,
    RESULTS_ROOT,
    git_head,
    load_development_inputs,
    load_target_scene,
    relative,
    write_csv,
    write_json,
)
from .pcms import select_pcms_candidates, select_point_target_candidates
from .provenance import sha256_file
from .sac import SACVariant, run_sac
from .smg import run_smg, sac_mapped_supported_micro_target
from .uatp import run_uatp_scene


REQUIRED_BASE_COMMIT = "7ffcbca365122a7acbf5232d515d829f1e27b8bd"
REQUIRED_PROTOCOL_HASH = "2e892f25b1c1770a55fa29e36206b634f424f9ff2508bf045c92d388e69b4be6"
REQUIRED_ABLATION_HASH = "e02cb6a22355fa66cd66c392e3199ccbb7508413e2db3b0f3977fc1796f0ccd6"
EXPECTED_FROZEN_TARGETS = {
    "bellevue_116th_ne12th": 10,
    "bellevue_150th_newport": 12,
    "bellevue_150th_eastgate": 9,
    "bellevue_150th_se38th": 18,
    "bellevue_ne8th": 9,
}


def utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={REPOSITORY_ROOT.as_posix()}",
            *arguments,
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
    ).strip()


def preflight() -> dict[str, Any]:
    """Verify every frozen scientific input without reading forbidden artifacts."""
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    actual_protocol = sha256_file(PROTOCOL_PATH)
    actual_ablation = sha256_file(ABLATION_PATH)
    merge_base = _git("merge-base", "HEAD", REQUIRED_BASE_COMMIT)
    checks = {
        "required_base_is_ancestor": merge_base == REQUIRED_BASE_COMMIT,
        "protocol_hash_matches": actual_protocol == REQUIRED_PROTOCOL_HASH,
        "ablation_hash_matches": actual_ablation == REQUIRED_ABLATION_HASH,
        "branch_matches": _git("branch", "--show-current")
        == "feature/futuretransp-hg-smg-development",
    }
    frozen_rows = []
    for name, item in protocol["frozen_inputs"].items():
        path = REPOSITORY_ROOT / item["path"]
        actual = sha256_file(path)
        frozen_rows.append(
            {
                "input_id": name,
                "path": item["path"],
                "expected_sha256": item["sha256"],
                "actual_sha256": actual,
                "matches": actual == item["sha256"],
            }
        )
    checks["all_frozen_inputs_match"] = all(row["matches"] for row in frozen_rows)
    checks["independent_test_guard_active"] = True
    checks["hg_smg_output_absent_at_initial_preflight"] = not RESULTS_ROOT.exists()
    if not all(checks.values()):
        raise RuntimeError(f"HG-SMG preflight failed: {checks}")
    payload = {
        "task": "Task 09B HG-SMG development preflight",
        "timestamp_utc": utc_timestamp(),
        "git_head": git_head(),
        "required_base_commit": REQUIRED_BASE_COMMIT,
        "required_protocol_sha256": REQUIRED_PROTOCOL_HASH,
        "actual_protocol_sha256": actual_protocol,
        "required_ablation_sha256": REQUIRED_ABLATION_HASH,
        "actual_ablation_sha256": actual_ablation,
        "checks": checks,
        "frozen_inputs": frozen_rows,
        "forbidden_data_accessed": False,
    }
    report_lines = [
        "# HG-SMG Development Preflight Report",
        "",
        f"- Timestamp: `{payload['timestamp_utc']}`",
        f"- Git HEAD at preflight: `{payload['git_head']}`",
        f"- Required base commit: `{REQUIRED_BASE_COMMIT}` (verified ancestor)",
        f"- Protocol SHA-256: `{actual_protocol}` (match)",
        f"- Ablation SHA-256: `{actual_ablation}` (match)",
        "- Independent-test access guard: **active**",
        "- Existing HG-SMG scientific outputs before implementation: **none**",
        "- Semantic/reference labels accessed: **no**",
        "",
        "## Frozen Inputs",
        "",
        "| Input | SHA-256 match | Repository-relative path |",
        "| --- | --- | --- |",
    ]
    report_lines.extend(
        f"| `{row['input_id']}` | {'yes' if row['matches'] else 'no'} | `{row['path']}` |"
        for row in frozen_rows
    )
    report_lines.extend(
        [
            "",
            "The preflight read only preregistered control files and hashes. It did not read independent-test assignments, metrics, polygon labels, scene guides, or movement inventories.",
        ]
    )
    path = PUBLICATION_ROOT / "docs/hg_smg_development_preflight_report.md"
    path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    write_json(payload, PUBLICATION_ROOT / "results/hg_smg/preflight.json")
    return payload


def _emd_endpoints(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        [
            "trajectory_id",
            "recording_id",
            "start_x_topview",
            "start_y_topview",
            "end_x_topview",
            "end_y_topview",
        ]
    ].copy()


def _full_scene_run(
    scene: str,
    scene_index: int,
    variants: tuple[SACVariant, ...],
) -> dict[str, Any]:
    inputs = load_development_inputs("target")
    frame, _, heading_provenance = load_target_scene(inputs, scene)
    target_config = inputs.runner_config["target_estimation"]
    frozen_seed = int(inputs.runner_config["random_seed"]) + scene_index * 1000
    emd = estimate_hg_target_detailed(
        scene,
        _emd_endpoints(frame),
        frozen_seed,
        [int(value) for value in target_config["region_counts"]],
        [float(value) for value in target_config["support_thresholds"]],
        int(target_config["region_metric_sample_size"]),
    )
    expected_assignments = pd.read_parquet(
        PUBLICATION_ROOT
        / "results/target_estimation/target_endpoint_region_assignments.parquet",
        filters=[("scene", "==", scene), ("split", "==", "target_estimation")],
        columns=["trajectory_id", "entry_region", "exit_region"],
    ).sort_values("trajectory_id", kind="mergesort")
    actual_assignments = pd.DataFrame(
        {
            "trajectory_id": frame["trajectory_id"],
            "entry_region": emd.entry_fit.labels,
            "exit_region": emd.exit_fit.labels,
        }
    ).sort_values("trajectory_id", kind="mergesort")
    labels_match = actual_assignments.reset_index(drop=True).equals(
        expected_assignments.reset_index(drop=True)
    )
    if not labels_match or int(emd.summary["hg_estimated_target"]) != EXPECTED_FROZEN_TARGETS[scene]:
        raise RuntimeError(f"{scene}: frozen EMD reproduction failed")
    reproduction = {
        "scene": scene,
        "frozen_target": EXPECTED_FROZEN_TARGETS[scene],
        "reproduced_target": int(emd.summary["hg_estimated_target"]),
        "entry_regions": int(emd.entry_fit.selected_count),
        "exit_regions": int(emd.exit_fit.selected_count),
        "support_threshold": float(emd.summary["support_threshold"]),
        "od_coverage": float(emd.summary["od_coverage"]),
        "entry_assignments_exact_match": labels_match,
        "exit_assignments_exact_match": labels_match,
        "reproduction_passed": True,
        "random_seed": frozen_seed,
    }
    descriptors = []
    pairs = []
    traces = []
    assignments = []
    supernodes = []
    edges = []
    thresholds = []
    targets = []
    a2_targets = []
    descriptor_cache: dict[tuple[object, ...], dict[str, Any]] = {}
    for variant in variants:
        sac = run_sac(
            scene,
            frame,
            emd.entry_fit.labels,
            emd.exit_fit.labels,
            emd.entry_fit.center,
            emd.exit_fit.center,
            variant,
            bootstrap_replicates=int(
                inputs.protocol["modules"]["SAC"]["self_consistency"][
                    "bootstrap_replicates"
                ]
            ),
            descriptor_cache=descriptor_cache,
        )
        smg = run_smg(
            scene,
            frame["trajectory_id"],
            emd.entry_fit.labels,
            emd.exit_fit.labels,
            sac,
            [float(value) for value in target_config["support_thresholds"]],
        )
        descriptors.append(sac.descriptors)
        pairs.append(sac.pairwise)
        traces.append(sac.merge_trace)
        assignments.append(sac.assignments)
        supernodes.append(sac.supernodes)
        edges.append(smg.edges)
        thresholds.append(smg.threshold_candidates)
        targets.append(smg.summary)
        if variant.variant_id == "A5":
            a2_targets.append(
                {
                    "scene": scene,
                    "variant_id": "A2",
                    "point_target": sac_mapped_supported_micro_target(
                        emd.od_counts,
                        float(emd.summary["support_threshold"]),
                        sac,
                    ),
                }
            )
    a8_status = {
        "scene": scene,
        "variant_id": "A8",
        "status": "not_identifiable_protocol_missing_jsd_threshold",
        "semantic_reference_access": False,
    }
    return {
        "reproduction": reproduction,
        "frame_provenance": {"scene": scene, **heading_provenance},
        "descriptors": pd.concat(descriptors, ignore_index=True),
        "pairs": pd.concat(pairs, ignore_index=True),
        "traces": pd.concat(traces, ignore_index=True) if any(len(x) for x in traces) else pd.DataFrame(),
        "assignments": pd.concat(assignments, ignore_index=True),
        "supernodes": pd.concat(supernodes, ignore_index=True),
        "edges": pd.concat(edges, ignore_index=True),
        "thresholds": pd.concat(thresholds, ignore_index=True),
        "targets": pd.DataFrame(targets),
        "a2_targets": pd.DataFrame(a2_targets),
        "a8_status": a8_status,
    }


def run_full_split(n_jobs: int = 1, output_root: Path = RESULTS_ROOT) -> None:
    inputs = load_development_inputs("target")
    scenes = tuple(inputs.protocol["scenes"]["fixed"])
    variants = ablations.executable_sac_variants()
    outputs = Parallel(n_jobs=int(n_jobs), backend="loky")(
        delayed(_full_scene_run)(scene, index, variants)
        for index, scene in enumerate(scenes)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    mappings = {
        "emd_reproduction.csv": "reproduction",
        "sac_microregion_descriptors.csv": "descriptors",
        "sac_pairwise_compatibility.csv": "pairs",
        "sac_merge_trace.csv": "traces",
        "sac_supernode_assignments.csv": "assignments",
        "sac_supernodes.csv": "supernodes",
        "smg_edges.csv": "edges",
        "smg_threshold_candidates.csv": "thresholds",
        "smg_targets.csv": "targets",
        "a2_targets.csv": "a2_targets",
    }
    for filename, key in mappings.items():
        values = [pd.DataFrame([item[key]]) if isinstance(item[key], dict) else item[key] for item in outputs]
        write_csv(pd.concat(values, ignore_index=True), output_root / filename)
    write_csv(
        pd.DataFrame([item["a8_status"] for item in outputs]),
        output_root / "a8_protocol_status.csv",
    )
    write_json(
        {item["frame_provenance"]["scene"]: item["frame_provenance"] for item in outputs},
        output_root / "development_input_provenance.json",
    )


def _uatp_scene_worker(
    scene: str,
    variants: tuple[SACVariant, ...],
    replicates: int,
    sac_replicates: int,
    replicate_jobs: int,
    full_split_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs = load_development_inputs("target")
    frame, _, _ = load_target_scene(inputs, scene)
    cfg = inputs.runner_config["target_estimation"]
    result = run_uatp_scene(
        scene,
        frame,
        variants,
        [int(value) for value in cfg["region_counts"]],
        [float(value) for value in cfg["support_thresholds"]],
        int(cfg["region_metric_sample_size"]),
        bootstrap_replicates=replicates,
        sac_bootstrap_replicates=sac_replicates,
        n_jobs=replicate_jobs,
    )
    full_targets = pd.read_csv(full_split_root / "smg_targets.csv")
    target_map = full_targets[full_targets["scene"] == scene].set_index("variant_id")["smg_target"]
    summary = result.summary.copy()
    summary["probability_full_split_target"] = [
        float(
            np.mean(
                result.runs.loc[result.runs["variant_id"] == variant, "smg_target"]
                == int(target_map[variant])
            )
        )
        for variant in summary["variant_id"]
    ]
    return result.runs, summary


def run_uatp(
    replicates: int = 500,
    sac_replicates: int = 500,
    n_jobs: int = 1,
    variant_ids: tuple[str, ...] = ("A5",),
    replicate_jobs: int = 1,
    output_root: Path = RESULTS_ROOT,
) -> None:
    inputs = load_development_inputs("target")
    variants = tuple(
        variant
        for variant in ablations.executable_sac_variants()
        if variant.variant_id in set(variant_ids)
    )
    if not variants:
        raise ValueError("No executable UATP variants selected")
    outputs = Parallel(n_jobs=int(n_jobs), backend="loky")(
        delayed(_uatp_scene_worker)(
            scene,
            variants,
            replicates,
            sac_replicates,
            replicate_jobs,
            output_root,
        )
        for scene in inputs.protocol["scenes"]["fixed"]
    )
    write_csv(
        pd.concat([item[0] for item in outputs], ignore_index=True),
        output_root / "uatp_bootstrap_targets.csv",
    )
    write_csv(
        pd.concat([item[1] for item in outputs], ignore_index=True),
        output_root / "uatp_summary.csv",
    )


def run_pcms(output_root: Path = RESULTS_ROOT) -> None:
    load_development_inputs("select")
    candidates = pd.read_csv(
        PUBLICATION_ROOT / "results/development/model_selection_candidates.csv"
    )
    summaries = pd.read_csv(output_root / "uatp_summary.csv")
    full_targets = pd.read_csv(output_root / "smg_targets.csv")
    primary = summaries[summaries["variant_id"] == "A5"].copy()
    intervals = primary.rename(
        columns={"interval_90_lower": "interval_lower", "interval_90_upper": "interval_upper"}
    )
    all_candidates, selected = select_pcms_candidates(candidates, intervals, "A5")
    write_csv(all_candidates, output_root / "pcms_candidates.csv")
    write_csv(selected, output_root / "pcms_selected_configurations.csv")
    selection_tables = [selected.assign(ablation_id="A5")]
    a2 = pd.read_csv(output_root / "a2_targets.csv")
    selection_tables.append(
        select_point_target_candidates(candidates, a2, "A2").assign(ablation_id="A2")
    )
    a3 = full_targets[full_targets["variant_id"] == "A5"][
        ["scene", "smg_target"]
    ].rename(columns={"smg_target": "point_target"})
    selection_tables.append(
        select_point_target_candidates(candidates, a3, "A3").assign(ablation_id="A3")
    )
    a10 = primary[["scene", "mode"]].rename(columns={"mode": "point_target"})
    selection_tables.append(
        select_point_target_candidates(candidates, a10, "A10").assign(ablation_id="A10")
    )
    for variant_id in ("A6", "A7", "A9"):
        variant_rows = summaries[summaries["variant_id"] == variant_id]
        if len(variant_rows):
            variant_intervals = variant_rows.rename(
                columns={
                    "interval_90_lower": "interval_lower",
                    "interval_90_upper": "interval_upper",
                }
            )
            _, variant_selected = select_pcms_candidates(
                candidates, variant_intervals, variant_id
            )
            selection_tables.append(variant_selected.assign(ablation_id=variant_id))
    selections = pd.concat(selection_tables, ignore_index=True)
    write_csv(selections, output_root / "ablation_selected_configurations.csv")
    provenance = {
        "protocol_sha256": REQUIRED_PROTOCOL_HASH,
        "candidate_input": relative(
            PUBLICATION_ROOT / "results/development/model_selection_candidates.csv"
        ),
        "candidate_input_sha256": sha256_file(
            PUBLICATION_ROOT / "results/development/model_selection_candidates.csv"
        ),
        "selection_order": "exact preregistered method-specific PCMS order",
        "emas_hg_used_for_selection": False,
        "reference_labels_accessed": False,
        "independent_test_accessed": False,
        "selected_count": int(len(selected)),
        "deterministic_provenance": True,
    }
    write_json(provenance, output_root / "pcms_selection_provenance.json")


def future_test_gate(confirmation: str | None) -> None:
    required = "I_CONFIRM_FROZEN_HG_SMG_EXTENSION_EVALUATION"
    freeze = PUBLICATION_ROOT / "configs/hg_smg_development_freeze_v1.sha256"
    if not freeze.exists():
        raise PermissionError("HG-SMG development freeze does not exist")
    if confirmation != required:
        raise PermissionError("Exact HG-SMG extension confirmation phrase is required")
    raise PermissionError(
        "Task 09B never permits independent-test execution; use a separate versioned task."
    )


def deterministic_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def run_determinism_check(
    replicates: int = 500,
    sac_replicates: int = 500,
    replicate_jobs: int = 1,
) -> dict[str, Any]:
    """Rerun the complete development pipeline in a clean second directory."""
    second_root = PUBLICATION_ROOT / "results/hg_smg/determinism_run2"
    resolved = second_root.resolve()
    allowed_parent = (PUBLICATION_ROOT / "results/hg_smg").resolve()
    if allowed_parent not in resolved.parents:
        raise RuntimeError(f"Unsafe determinism output path: {resolved}")
    if second_root.exists():
        shutil.rmtree(second_root)
    variants = ("A5", "A6", "A7", "A9")
    run_full_split(n_jobs=3, output_root=second_root)
    run_uatp(
        replicates=replicates,
        sac_replicates=sac_replicates,
        n_jobs=1,
        variant_ids=variants,
        replicate_jobs=replicate_jobs,
        output_root=second_root,
    )
    run_pcms(output_root=second_root)
    compared = (
        "sac_supernode_assignments.csv",
        "smg_edges.csv",
        "uatp_bootstrap_targets.csv",
        "uatp_summary.csv",
        "pcms_selected_configurations.csv",
        "pcms_selection_provenance.json",
    )
    rows = []
    for name in compared:
        first = sha256_file(RESULTS_ROOT / name)
        second = sha256_file(second_root / name)
        rows.append(
            {
                "artifact": name,
                "run1_sha256": first,
                "run2_sha256": second,
                "matches": first == second,
            }
        )
    table = pd.DataFrame(rows)
    write_csv(table, RESULTS_ROOT / "determinism_hash_comparison.csv")
    payload = {
        "complete_pipeline_rerun": True,
        "replicates": int(replicates),
        "sac_bootstrap_replicates": int(sac_replicates),
        "all_compared_artifacts_match": bool(table["matches"].all()),
        "compared_artifact_count": int(len(table)),
        "second_run_directory": relative(second_root),
        "independent_test_access": False,
        "reference_label_access": False,
    }
    if not payload["all_compared_artifacts_match"]:
        raise RuntimeError("HG-SMG development determinism check failed")
    write_json(payload, RESULTS_ROOT / "determinism_summary.json")
    return payload
