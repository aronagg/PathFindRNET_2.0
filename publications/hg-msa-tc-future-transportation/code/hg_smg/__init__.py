"""Development-only implementation of the preregistered HG-SMG-TC extension."""

from .provenance import IMPLEMENTATION_VERSION, MASTER_SEED, derive_seed

__all__ = ["IMPLEMENTATION_VERSION", "MASTER_SEED", "derive_seed"]
