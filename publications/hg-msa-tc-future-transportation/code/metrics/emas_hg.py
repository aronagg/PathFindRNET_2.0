"""Canonical implementation of the frozen task-specific EMAS_HG-v1 score.

This module formalizes the score already used by the frozen HG-MSA-TC experiments.
It does not define or optimize alternative scientific protocols.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, isnan
from typing import Any, Mapping


EMAS_HG_VERSION = "EMAS_HG-v1"


@dataclass(frozen=True)
class EmasWeights:
    """Non-negative EMAS component weights that sum to one."""

    target: float
    outlier: float
    balance: float
    silhouette: float
    davies_bouldin: float

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        """Return weights in canonical T, O, B, S, D order."""
        return (
            float(self.target),
            float(self.outlier),
            float(self.balance),
            float(self.silhouette),
            float(self.davies_bouldin),
        )

    def as_dict(self) -> dict[str, float]:
        """Return weights with concise component names."""
        return dict(zip(("T", "O", "B", "S", "D"), self.as_tuple(), strict=True))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> EmasWeights:
        """Build weights from either concise or descriptive keys."""
        aliases = {
            "target": ("T", "target"),
            "outlier": ("O", "outlier"),
            "balance": ("B", "balance"),
            "silhouette": ("S", "silhouette"),
            "davies_bouldin": ("D", "davies_bouldin"),
        }
        output: dict[str, float] = {}
        for field, keys in aliases.items():
            found = [key for key in keys if key in values]
            if len(found) != 1:
                raise ValueError(f"Exactly one weight key is required for {field}: {keys}")
            output[field] = float(values[found[0]])
        weights = cls(**output)
        weights.validate()
        return weights

    def validate(self, tolerance: float = 1e-12) -> None:
        """Reject non-finite, negative, or non-unit-sum weights."""
        values = self.as_tuple()
        if any(not isfinite(value) for value in values):
            raise ValueError("EMAS weights must be finite.")
        if any(value < 0.0 for value in values):
            raise ValueError("EMAS weights must be non-negative.")
        if abs(sum(values) - 1.0) > tolerance:
            raise ValueError(f"EMAS weights must sum to one; received {sum(values):.17g}.")


ORIGINAL_WEIGHTS = EmasWeights(0.50, 0.20, 0.10, 0.10, 0.10)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return isnan(float(value))
    except (TypeError, ValueError):
        return False


def _clip_unit(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def validate_emas_inputs(
    *,
    expected_target: Any,
    cluster_count_error: Any,
    pct_outliers: Any,
    largest_cluster_ratio: Any = None,
    silhouette_clustered_only: Any = None,
    davies_bouldin_clustered_only: Any = None,
    weights: EmasWeights = ORIGINAL_WEIGHTS,
) -> None:
    """Validate the score's required contract without altering optional fallbacks.

    Zero is accepted as a legacy target and uses an effective denominator of one.
    Missing targets, count errors, and outlier percentages are invalid. Optional
    internal metrics may be missing because EMAS_HG-v1 assigns them a neutral 0.5.
    """
    weights.validate()
    required = {
        "expected_target": expected_target,
        "cluster_count_error": cluster_count_error,
        "pct_outliers": pct_outliers,
    }
    for name, value in required.items():
        if _is_missing(value):
            raise ValueError(f"{name} is required for {EMAS_HG_VERSION}.")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be numeric.") from exc
        if not isfinite(numeric):
            raise ValueError(f"{name} must be finite.")
    if float(expected_target) < 0.0:
        raise ValueError("expected_target must be non-negative.")
    if float(cluster_count_error) < 0.0:
        raise ValueError("cluster_count_error must be non-negative.")
    for name, value in {
        "largest_cluster_ratio": largest_cluster_ratio,
        "silhouette_clustered_only": silhouette_clustered_only,
        "davies_bouldin_clustered_only": davies_bouldin_clustered_only,
    }.items():
        if not _is_missing(value):
            try:
                float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{name} must be numeric or missing.") from exc


def target_agreement_component(expected_target: Any, cluster_count_error: Any) -> float:
    """Return T = clip(1 - absolute count error / max(target, 1), 0, 1)."""
    denominator = max(float(expected_target), 1.0)
    return _clip_unit(1.0 - float(cluster_count_error) / denominator)


def non_outlier_component(pct_outliers: Any) -> float:
    """Return O = clip(1 - outlier percentage / 100, 0, 1)."""
    return _clip_unit(1.0 - float(pct_outliers) / 100.0)


def cluster_balance_component(largest_cluster_ratio: Any) -> float:
    """Return B = clip(1 - largest clustered-only share, 0, 1).

    The frozen implementation uses 0.5 when the ratio is missing. Noise is excluded
    upstream from both the largest-cluster numerator and clustered-only denominator.
    """
    if _is_missing(largest_cluster_ratio):
        return 0.5
    return _clip_unit(1.0 - float(largest_cluster_ratio))


def silhouette_component(silhouette_clustered_only: Any) -> float:
    """Return S = clip((silhouette + 1) / 2, 0, 1), or 0.5 if undefined."""
    if _is_missing(silhouette_clustered_only):
        return 0.5
    return _clip_unit((float(silhouette_clustered_only) + 1.0) / 2.0)


def davies_bouldin_component(davies_bouldin_clustered_only: Any) -> float:
    """Return D = 1 / (1 + DB) for non-negative DB, or 0.5 if invalid.

    Positive infinity maps to zero, matching the frozen expression. Missing,
    negative, and negative-infinite values use the legacy neutral fallback 0.5.
    """
    if _is_missing(davies_bouldin_clustered_only):
        return 0.5
    value = float(davies_bouldin_clustered_only)
    if value < 0.0:
        return 0.5
    return 1.0 / (1.0 + value)


def compute_emas_hg(
    *,
    expected_target: Any,
    cluster_count_error: Any,
    pct_outliers: Any,
    largest_cluster_ratio: Any = None,
    silhouette_clustered_only: Any = None,
    davies_bouldin_clustered_only: Any = None,
    weights: EmasWeights = ORIGINAL_WEIGHTS,
) -> float:
    """Compute the frozen task-specific EMAS_HG-v1 weighted score."""
    validate_emas_inputs(
        expected_target=expected_target,
        cluster_count_error=cluster_count_error,
        pct_outliers=pct_outliers,
        largest_cluster_ratio=largest_cluster_ratio,
        silhouette_clustered_only=silhouette_clustered_only,
        davies_bouldin_clustered_only=davies_bouldin_clustered_only,
        weights=weights,
    )
    components = (
        target_agreement_component(expected_target, cluster_count_error),
        non_outlier_component(pct_outliers),
        cluster_balance_component(largest_cluster_ratio),
        silhouette_component(silhouette_clustered_only),
        davies_bouldin_component(davies_bouldin_clustered_only),
    )
    score = sum(weight * component for weight, component in zip(weights.as_tuple(), components))
    return float(_clip_unit(score))
