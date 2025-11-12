import os

import hydra
from omegaconf import DictConfig

from traffic.io.dataset_loader import get_paths
from traffic.io.legacy_io import load_legacy_json
from traffic.io.serialization import write_parquet


@hydra.main(config_path="../configs", config_name="defaults", version_base=None)
def main(cfg: DictConfig):
    """Import a legacy entities JSON and write tracks.parquet under interim/.

    Usage examples (run from repo root):
      - python scripts/import_legacy_json.py input_json=/absolute/path/to/entities.json
      - python scripts/import_legacy_json.py input_json=data/raw/sample/entities.json
      - Or set dataset.legacy_json in a config and call without input_json=...
    """
    # logger = logging.getLogger(__name__)
    # if not logger.handlers:
    #     handler = logging.StreamHandler()
    #     handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    #     logger.addHandler(handler)
    # logger.setLevel(logging.DEBUG)

    # # Log full cfg (resolved) and dataset keys (if present) for debugging
    # logger.debug("cfg: %s", OmegaConf.to_container(cfg, resolve=True))
    # if hasattr(cfg, "dataset") and cfg.dataset is not None:
    #     try:
    #         dataset_keys = list(cfg.dataset.keys())
    #     except Exception:
    #         dataset_keys = []
    #     logger.debug("dataset keys: %s", dataset_keys)
    # else:
    #     logger.debug("cfg has no 'dataset' key")
    raw, interim, _ = get_paths(cfg.dataset)

    # Resolve input file
    source = getattr(cfg, "input_json", None)
    if source is None:
        # Optional config location (relative to original CWD)
        source = getattr(cfg.dataset, "legacy_json", None)
        if source is None:
            raise SystemExit(
                "Provide input_json=<path> on the CLI or set dataset.legacy_json in your config"
            )
    # Make relative paths resolve from original working directory (not the Hydra run dir)
    if not os.path.isabs(source):
        source = os.path.join(hydra.utils.get_original_cwd(), source)

    # Optional class map: pick from top-level or dataset scope if provided
    class_map = getattr(cfg, "class_map", None)
    if class_map is None and hasattr(cfg, "dataset"):
        class_map = getattr(cfg.dataset, "class_map", None)

    df = load_legacy_json(source, class_map=class_map)

    out = interim / "tracks.parquet"
    write_parquet(df, out)
    print(f"Imported {len(df)} rows from {source} -> {out}")


if __name__ == "__main__":
    main()
