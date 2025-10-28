from pathlib import Path

from omegaconf import DictConfig


def get_paths(cfg: DictConfig):
    raw = Path(cfg.raw_dir)
    interim = Path(cfg.interim_dir)
    processed = Path(cfg.processed_dir)
    raw.mkdir(parents=True, exist_ok=True)
    interim.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    return raw, interim, processed
