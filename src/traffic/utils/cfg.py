from omegaconf import DictConfig


def asdict(cfg: DictConfig):
    return {k: cfg[k] for k in cfg.keys()}
