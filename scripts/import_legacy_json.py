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
      - python scripts/import_legacy_json.py source=/absolute/path/to/entities.json
      - python scripts/import_legacy_json.py source=data/raw/sample/entities.json
      - Or set dataset.legacy_json in a config and call without source=...
    """
    raw, interim, _ = get_paths(cfg.dataset)

    # Resolve input file
    source = getattr(cfg, "source", None)
    if source is None:
        # Optional config location (relative to original CWD)
        source = getattr(cfg.dataset, "legacy_json", None)
        if source is None:
            raise SystemExit(
                "Provide source=<path> on the CLI or set dataset.legacy_json in your config"
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
