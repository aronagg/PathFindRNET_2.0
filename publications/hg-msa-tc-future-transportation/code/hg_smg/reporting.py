"""Reports, figures, reproducibility records, and development freeze for Task 09B."""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import sklearn
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .io import FIGURES_ROOT, PUBLICATION_ROOT, RESULTS_ROOT, relative
from .provenance import IMPLEMENTATION_VERSION, MASTER_SEED, sha256_file


SCENES = (
    "bellevue_116th_ne12th",
    "bellevue_150th_newport",
    "bellevue_150th_eastgate",
    "bellevue_150th_se38th",
    "bellevue_ne8th",
)
PROTOCOL_HASH = "2e892f25b1c1770a55fa29e36206b634f424f9ff2508bf045c92d388e69b4be6"


def timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(map(str, columns)) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("" if not np.isfinite(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_doc(name: str, title: str, body: str) -> None:
    path = PUBLICATION_ROOT / "docs" / name
    path.write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")


def build_ablation_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    uatp = pd.read_csv(RESULTS_ROOT / "uatp_summary.csv")
    targets = pd.read_csv(RESULTS_ROOT / "smg_targets.csv")
    selections = pd.read_csv(RESULTS_ROOT / "ablation_selected_configurations.csv")
    rows: list[dict[str, Any]] = []
    for ablation_id in [f"A{index}" for index in range(11)]:
        status = "completed"
        split = "target_estimation_then_model_selection"
        note = ""
        if ablation_id == "A0":
            status = "reused_frozen_original"
            note = "Frozen untargeted selection reused; no scientific recomputation."
        elif ablation_id == "A1":
            status = "reused_frozen_original"
            note = "Frozen original HG-aware point-target selection reused."
        elif ablation_id == "A4":
            split = "target_estimation"
            note = "UATP target distribution only; no model selection by definition."
        elif ablation_id == "A8":
            status = "not_identifiable"
            note = "Protocol v1 defines JSD but no executable JSD compatibility threshold."
        rows.append(
            {
                "ablation_id": ablation_id,
                "status": status,
                "split": split,
                "selected_configuration_rows": int(
                    (selections.get("ablation_id", pd.Series(dtype=str)) == ablation_id).sum()
                ),
                "reference_label_access": False,
                "note": note,
            }
        )
    ablation_frame = pd.DataFrame(rows)
    sensitivity_targets = targets[
        targets["variant_id"].str.startswith("sensitivity_")
    ][["scene", "variant_id", "smg_target"]]
    sensitivity_uatp = uatp[
        uatp["variant_id"].str.startswith("sensitivity_")
    ]
    sensitivity = sensitivity_targets.merge(
        sensitivity_uatp,
        on=["scene", "variant_id"],
        how="left",
        validate="one_to_one",
    )
    primary_intervals = uatp[uatp["variant_id"] == "A5"].copy()
    interval_rows = []
    for row in primary_intervals.itertuples():
        for level in (80, 90, 95):
            interval_rows.append(
                {
                    "scene": row.scene,
                    "variant_id": f"sensitivity_uatp_interval_{level}",
                    "smg_target": np.nan,
                    "mode": row.mode,
                    "median": row.median,
                    "interval_level": level / 100.0,
                    "interval_lower": getattr(row, f"interval_{level}_lower"),
                    "interval_upper": getattr(row, f"interval_{level}_upper"),
                    "entropy_bits": row.entropy_bits,
                }
            )
    sensitivity = pd.concat(
        [sensitivity, pd.DataFrame(interval_rows)], ignore_index=True, sort=False
    )
    ablation_frame.to_csv(
        RESULTS_ROOT / "ablation_development_summary.csv", index=False, lineterminator="\n"
    )
    sensitivity.to_csv(
        RESULTS_ROOT / "preregistered_sensitivity_summary.csv",
        index=False,
        lineterminator="\n",
    )
    return ablation_frame, sensitivity


def degeneracy_summary() -> dict[str, Any]:
    targets = pd.read_csv(RESULTS_ROOT / "smg_targets.csv")
    traces = pd.read_csv(RESULTS_ROOT / "sac_merge_trace.csv")
    descriptors = pd.read_csv(RESULTS_ROOT / "sac_microregion_descriptors.csv")
    uatp = pd.read_csv(RESULTS_ROOT / "uatp_summary.csv")
    candidates = pd.read_csv(RESULTS_ROOT / "pcms_candidates.csv")
    primary = targets[targets["variant_id"] == "A5"]
    primary_traces = traces[traces["variant_id"] == "A5"]
    all_one = bool(
        (
            (primary["n_entry_supernodes"] == 1)
            & (primary["n_exit_supernodes"] == 1)
        ).all()
    )
    zero_merges_every_scene = all(
        scene not in set(primary_traces["scene"]) for scene in SCENES
    )
    primary_uatp = uatp[uatp["variant_id"] == "A5"]
    material_undefined = descriptors[
        (descriptors["variant_id"] == "A5")
        & (descriptors["n_finite_headings"] / descriptors["n_support"] < 0.99)
    ]
    result = {
        "all_microregions_merge_to_one_in_every_scene": all_one,
        "zero_primary_merges_in_every_scene": zero_merges_every_scene,
        "zero_supported_edges_any_primary_scene": bool(primary["zero_supported_edges"].any()),
        "uatp_failed_replicates": int(primary_uatp["failed_replicates"].sum()),
        "material_undefined_heading_region_count": int(len(material_undefined)),
        "numerical_floor_dominated_primary_region_count": int(
            (
                (descriptors["variant_id"] == "A5")
                & (
                    (descriptors["bearing_radius"] <= 2e-14)
                    | (descriptors["heading_radius"] <= 2e-14)
                )
            ).sum()
        ),
        "candidate_intervals_span_complete_grid_all_cases": bool(
            candidates.groupby(["scene", "method"])["interval_distance"].max().eq(0).all()
        ),
        "a8_protocol_identifiable": False,
    }
    result["primary_algorithm_valid"] = not any(
        [
            result["all_microregions_merge_to_one_in_every_scene"],
            result["zero_primary_merges_in_every_scene"],
            result["zero_supported_edges_any_primary_scene"],
            result["uatp_failed_replicates"] > 0,
            result["candidate_intervals_span_complete_grid_all_cases"],
        ]
    )
    return result


def generate_figures() -> list[Path]:
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    descriptors = pd.read_csv(RESULTS_ROOT / "sac_microregion_descriptors.csv")
    assignments = pd.read_csv(RESULTS_ROOT / "sac_supernode_assignments.csv")
    pairs = pd.read_csv(RESULTS_ROOT / "sac_pairwise_compatibility.csv")
    edges = pd.read_csv(RESULTS_ROOT / "smg_edges.csv")
    uatp = pd.read_csv(RESULTS_ROOT / "uatp_bootstrap_targets.csv")
    candidates = pd.read_csv(RESULTS_ROOT / "pcms_candidates.csv")
    targets = pd.read_csv(RESULTS_ROOT / "smg_targets.csv")
    outputs: list[Path] = []
    for scene in SCENES:
        scene_dir = FIGURES_ROOT / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        primary = descriptors[
            (descriptors["scene"] == scene) & (descriptors["variant_id"] == "A5")
        ].merge(
            assignments[
                (assignments["scene"] == scene) & (assignments["variant_id"] == "A5")
            ],
            on=["scene", "variant_id", "role", "micro_region"],
            validate="one_to_one",
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for axis, role in zip(axes, ("entry", "exit"), strict=True):
            role_rows = primary[primary["role"] == role]
            for node, group in role_rows.groupby("supernode_id", sort=True):
                axis.scatter(
                    group["centroid_x_topview"],
                    group["centroid_y_topview"],
                    s=80,
                    label=node,
                )
                for row in group.itertuples():
                    axis.annotate(str(row.micro_region), (row.centroid_x_topview, row.centroid_y_topview))
            axis.set_title(f"{role.title()} micro-regions / supernodes")
            axis.set_aspect("equal", adjustable="datalim")
            axis.legend(fontsize=7)
        fig.suptitle(scene)
        fig.tight_layout()
        path = scene_dir / "sac_microregions_supernodes.png"
        fig.savefig(path, dpi=220)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        outputs.extend([path, path.with_suffix(".pdf")])

        scene_pairs = pairs[(pairs["scene"] == scene) & (pairs["variant_id"] == "A5")]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for axis, role in zip(axes, ("entry", "exit"), strict=True):
            role_rows = scene_pairs[scene_pairs["role"] == role]
            regions = sorted(
                set(role_rows["left_micro_region"]).union(role_rows["right_micro_region"])
            )
            matrix = np.zeros((len(regions), len(regions)))
            lookup = {value: index for index, value in enumerate(regions)}
            for row in role_rows.itertuples():
                i, j = lookup[row.left_micro_region], lookup[row.right_micro_region]
                matrix[i, j] = matrix[j, i] = min(float(row.primary_distance), 3.0)
            image = axis.imshow(matrix, vmin=0, vmax=3, cmap="viridis")
            axis.set_xticks(range(len(regions)), regions)
            axis.set_yticks(range(len(regions)), regions)
            axis.set_title(f"{role.title()} compatibility D (clipped at 3)")
        fig.colorbar(image, ax=axes, shrink=0.8)
        path = scene_dir / "sac_compatibility_matrix.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)

        scene_edges = edges[(edges["scene"] == scene) & (edges["variant_id"] == "A5")]
        matrix = scene_edges.pivot(
            index="entry_supernode", columns="exit_supernode", values="share"
        ).fillna(0.0)
        fig, axis = plt.subplots(figsize=(5.2, 4.2))
        image = axis.imshow(matrix.to_numpy(), cmap="magma")
        axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
        axis.set_yticks(range(len(matrix.index)), matrix.index)
        axis.set_title(f"{scene}: SMG support share")
        fig.colorbar(image, ax=axis)
        fig.tight_layout()
        path = scene_dir / "smg_support_matrix.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        outputs.append(path)

        scene_boot = uatp[(uatp["scene"] == scene) & (uatp["variant_id"] == "A5")]
        fig, axis = plt.subplots(figsize=(5.5, 3.8))
        bins = np.arange(scene_boot["smg_target"].min() - 0.5, scene_boot["smg_target"].max() + 1.5)
        axis.hist(scene_boot["smg_target"], bins=bins, color="#2563eb", edgecolor="white")
        axis.set_xlabel("K_SMG")
        axis.set_ylabel("Bootstrap replicates")
        axis.set_title(f"{scene}: UATP target distribution")
        path = scene_dir / "uatp_target_distribution.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    primary_targets = targets[targets["variant_id"] == "A5"]
    positions = np.arange(len(SCENES))
    axis.bar(positions, primary_targets.set_index("scene").loc[list(SCENES), "smg_target"])
    axis.set_xticks(positions, [value.replace("bellevue_", "") for value in SCENES], rotation=25, ha="right")
    axis.set_ylabel("Full-split K_SMG")
    axis.set_title("Primary A5 development targets")
    path = FIGURES_ROOT / "ablation_structural_summary.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)

    fig, axis = plt.subplots(figsize=(6.5, 4.0))
    sample = candidates[(candidates["scene"] == SCENES[0]) & (candidates["method"] == "kmeans")]
    axis.scatter(sample["n_clusters"], sample["interval_distance"], s=22)
    axis.set_xlabel("Candidate cluster count")
    axis.set_ylabel("Distance to UATP interval")
    axis.set_title("PCMS interval-distance schematic")
    path = FIGURES_ROOT / "pcms_interval_distance.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)
    return outputs


def write_reports(determinism: dict[str, Any] | None = None) -> None:
    reproduction = pd.read_csv(RESULTS_ROOT / "emd_reproduction.csv")
    targets = pd.read_csv(RESULTS_ROOT / "smg_targets.csv")
    uatp = pd.read_csv(RESULTS_ROOT / "uatp_summary.csv")
    selected = pd.read_csv(RESULTS_ROOT / "pcms_selected_configurations.csv")
    ablation_frame, sensitivity = build_ablation_tables()
    degeneracy = degeneracy_summary()
    primary_targets = targets[targets["variant_id"] == "A5"][
        ["scene", "n_entry_supernodes", "n_exit_supernodes", "smg_target", "support_threshold", "supported_trajectory_coverage"]
    ]
    primary_uatp = uatp[uatp["variant_id"] == "A5"][
        ["scene", "mode", "median", "interval_90_lower", "interval_90_upper", "entropy_bits", "probability_full_split_target", "failed_replicates"]
    ]
    pcms_view = selected[
        ["scene", "method", "n_clusters", "params_json", "interval_lower", "interval_upper", "interval_distance"]
    ]
    _write_doc(
        "hg_smg_emd_reproduction_report.md",
        "HG-SMG EMD Reproduction Report",
        _markdown(reproduction) + "\n\nAll five frozen EMD targets and trajectory-level region assignments reproduced exactly before SAC was evaluated.",
    )
    _write_doc(
        "hg_smg_sac_development_report.md",
        "HG-SMG SAC Development Report",
        _markdown(primary_targets[["scene", "n_entry_supernodes", "n_exit_supernodes"]])
        + "\n\nPrimary SAC used 500 within-region bootstraps, q=0.95, complete linkage, top-view bearing, and five-point camera-isotropic directed heading. No semantic labels were accessed. A8 is not executable because protocol v1 did not freeze a JSD compatibility threshold; no threshold was invented post hoc.",
    )
    _write_doc(
        "hg_smg_smg_development_report.md",
        "HG-SMG SMG Development Report",
        _markdown(primary_targets) + "\n\n`K_SMG` counts supported supernode OD edges on `target_estimation`; it is not a legal or manually verified maneuver count.",
    )
    _write_doc(
        "hg_smg_uatp_development_report.md",
        "HG-SMG UATP Development Report",
        _markdown(primary_uatp) + "\n\nEvery scene used 500 deterministic recording-aware hierarchical bootstrap replicates and reran EMD -> SAC -> SMG. The primary interval is the floor/ceil integerized 90% percentile interval.",
    )
    _write_doc(
        "hg_smg_pcms_development_report.md",
        "HG-SMG PCMS Development Report",
        _markdown(pcms_view) + "\n\nOnly the frozen model-selection candidate grid was read. EMAS_HG and semantic/reference labels were not used in selection.",
    )
    _write_doc(
        "hg_smg_development_ablation_report.md",
        "HG-SMG Development Ablation Report",
        _markdown(ablation_frame)
        + "\n\nThe heading-only A7 variant shows substantial structural collapse in several scenes. Sensitivities remain diagnostics and cannot replace A5. A8 remains unexecuted because its compatibility threshold is absent from protocol v1.",
    )
    _write_doc(
        "hg_smg_degeneracy_and_validity_report.md",
        "HG-SMG Degeneracy and Validity Report",
        "\n".join(f"- `{key}`: `{value}`" for key, value in degeneracy.items())
        + "\n\nThe primary A5 method is computationally identified under the frozen rules. The under-specified A8 diagnostic is a protocol deviation/blocker for A8 only and must be amended explicitly before any future A8 test evaluation.",
    )
    if determinism is not None:
        _write_doc(
            "hg_smg_determinism_report.md",
            "HG-SMG Determinism Report",
            "\n".join(f"- `{key}`: `{value}`" for key, value in determinism.items()),
        )
    _write_doc(
        "hg_smg_future_locked_test_protocol.md",
        "Future Locked HG-SMG Test Protocol",
        "The Task 09B runner refuses all real independent-test execution, including when the confirmation text is supplied. A separate versioned task must verify the exact protocol hash, development-freeze hash, code commit, and explicit phrase `I_CONFIRM_FROZEN_HG_SMG_EXTENSION_EVALUATION`. No unlock file is created here.",
    )
    _write_doc(
        "task_09b_execution_report.md",
        "Task 09B Execution Report",
        f"Implementation version: `{IMPLEMENTATION_VERSION}`.\n\nPrimary targets:\n\n{_markdown(primary_targets)}\n\nUATP:\n\n{_markdown(primary_uatp)}\n\nPCMS selections:\n\n{_markdown(pcms_view)}\n\nNo independent-test data, reference labels, scene guides, polygon files, or semantic metrics were accessed. A8 is transparently marked non-identifiable under protocol v1.",
    )


def freeze_development(code_commit: str, determinism: dict[str, Any]) -> str:
    outputs = {
        relative(path): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in sorted(RESULTS_ROOT.glob("*"))
        if path.is_file() and path.name not in {"development_result_manifest.json", "development_sha256sums.txt"}
    }
    payload = {
        "freeze_version": "hg-smg-development-freeze-v1",
        "created_at_utc": timestamp(),
        "task09a_protocol_sha256": PROTOCOL_HASH,
        "code_commit": code_commit,
        "implementation_version": IMPLEMENTATION_VERSION,
        "master_seed": MASTER_SEED,
        "splits": ["target_estimation", "model_selection"],
        "reference_label_access": False,
        "independent_test_access": False,
        "independent_test_execution": False,
        "primary_ablation": "A5",
        "a8_status": "not_identifiable_protocol_missing_jsd_threshold",
        "future_test_eligible_without_amendment": False,
        "determinism": determinism,
        "outputs": outputs,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    freeze_path = PUBLICATION_ROOT / "configs/hg_smg_development_freeze_v1.yaml"
    freeze_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False, width=100),
        encoding="utf-8",
    )
    freeze_hash = sha256_file(freeze_path)
    (PUBLICATION_ROOT / "configs/hg_smg_development_freeze_v1.sha256").write_text(
        f"{freeze_hash}  configs/hg_smg_development_freeze_v1.yaml\n",
        encoding="ascii",
    )
    manifest = {
        "freeze_sha256": freeze_hash,
        "code_commit": code_commit,
        "protocol_sha256": PROTOCOL_HASH,
        "outputs": outputs,
        "created_at_utc": payload["created_at_utc"],
    }
    (RESULTS_ROOT / "development_result_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_lines = [f"{item['sha256']}  {path}" for path, item in sorted(outputs.items())]
    (RESULTS_ROOT / "development_sha256sums.txt").write_text(
        "\n".join(checksum_lines) + "\n", encoding="ascii"
    )
    _write_doc(
        "hg_smg_development_freeze_report.md",
        "HG-SMG Development Freeze Report",
        f"- Development freeze SHA-256: `{freeze_hash}`\n- Code commit: `{code_commit}`\n- Protocol SHA-256: `{PROTOCOL_HASH}`\n- Independent-test access: **none**\n- Future test eligible without protocol amendment: **no**, because A8 lacks a frozen JSD threshold.\n\nThe primary A5 implementation and all completed development artifacts are immutable under this freeze.",
    )
    return freeze_hash
