from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_final_method_comparison_contains_required_method_families() -> None:
    table = pd.read_csv(ROOT / "results" / "final_synthesis" / "final_method_comparison.csv")
    required = {
        "original_untargeted",
        "original_hg_aware_A1",
        "hg_smg_tc_A5",
        "endpoint_camera_raw",
        "endpoint_camera_isotropic",
        "resampled_trajectory_euclidean",
    }
    assert required.issubset(set(table["method_family"]))
    aggregate = table[(table["scene_id"] == "ALL_SCENES") & (table["method"] == "ALL_METHODS")]
    assert set(aggregate["method_family"]) == required


def test_final_method_comparison_metric_columns_are_complete() -> None:
    table = pd.read_csv(ROOT / "results" / "final_synthesis" / "final_method_comparison.csv")
    required_metrics = {
        "observed_target_abs_error",
        "legal_target_abs_error",
        "n_non_noise_clusters",
        "noise_pct_all_test",
        "ari",
        "nmi",
        "purity",
        "homogeneity",
        "completeness",
        "v_measure",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    }
    assert required_metrics.issubset(set(table.columns))
    assert table[list(required_metrics)].notna().all().all()


def test_paired_scene_differences_use_five_scene_unit() -> None:
    diff = pd.read_csv(ROOT / "results" / "final_synthesis" / "paired_scene_differences.csv")
    expected_comparisons = {
        "A1_minus_A0",
        "A5_minus_A1",
        "A5_minus_strongest_endpoint",
        "A5_minus_resampled_trajectory",
    }
    assert set(diff["comparison_id"]) == expected_comparisons
    counts = diff.groupby("comparison_id")["scene_id"].nunique()
    assert (counts == 5).all()


def test_task_12_did_not_create_zip_package() -> None:
    assert not (ROOT.parents[1] / "futuretransp_revision_12_final_results_synthesis.zip").exists()
    assert not (ROOT.parents[1] / "futuretransp_revision_12_final_results_synthesis.zip.sha256").exists()
