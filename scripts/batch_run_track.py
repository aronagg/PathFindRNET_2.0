import argparse
import glob
import subprocess
import sys
from pathlib import Path
from typing import List
import hydra
from hydra import initialize, compose
import hydra.utils
from omegaconf import OmegaConf


def collect_sources(path: Path, pattern: str, recursive: bool) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise SystemExit(f"Source not found: {path}")
    if recursive:
        return sorted(path.rglob(pattern))
    return sorted(path.glob(pattern))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", "-s", required=False, help="file or directory with videos (optional; if omitted the config's dataset.video is used)")
    p.add_argument("--config-name", default="defaults", help="hydra config name to pass to run_track (or path to a config file)")
    p.add_argument("--pattern", default="*.mp4", help="glob pattern for video files")
    p.add_argument("--recursive", action="store_true", help="search directories recursively")
    p.add_argument("--visualize", action="store_true", help="enable visualization for each run (adds +dataset.visualize=true)")
    p.add_argument("overrides", nargs="*", help="extra hydra overrides (e.g. +detect.device=0)")
    args = p.parse_args()

    # prepare overrides for composing config (include visualize if requested)
    compose_overrides = list(args.overrides)
    if args.visualize:
        compose_overrides.append("dataset.visualize=true")

    # determine input files: CLI --source wins; otherwise read dataset.video from config
    if args.source:
        src_path = Path(args.source)
        files = collect_sources(src_path, args.pattern, args.recursive)
    else:
        # load/compose config. support two cases:
        # 1) user passed a config name present under ../configs -> use hydra.compose
        # 2) user passed an arbitrary file path -> load with OmegaConf
        configs_dir = Path(__file__).parents[1] / "configs"
        cfg = None
        cfg_arg_path = Path(args.config_name)
        if cfg_arg_path.is_file():
            # if file lives under repo configs/, prefer composing (so defaults are resolved)
            try:
                cfg_arg_path.resolve().relative_to(configs_dir.resolve())
            except Exception:
                # arbitrary file -> load directly
                loaded = OmegaConf.load(str(cfg_arg_path))
                if "dataset" in loaded:
                    cfg = loaded
                else:
                    cfg = OmegaConf.create({"dataset": loaded})
            else:
                # file is inside configs/ -> compose by stem name
                cfg_name = cfg_arg_path.stem
                with initialize(version_base=None, config_path="../configs"):
                    cfg = compose(config_name=cfg_name, overrides=compose_overrides)
        else:
            # treat as hydra config name
            with initialize(version_base=None, config_path="../configs"):
                cfg = compose(config_name=args.config_name, overrides=compose_overrides)

        video_val = str(cfg.dataset.get("video"))
        if not Path(video_val).is_absolute():
            # run outside Hydra: use current working directory instead of hydra.utils.get_original_cwd()
            video_val = str(Path.cwd() / video_val)
        files = collect_sources(Path(video_val), args.pattern, args.recursive)

    if not files:
        raise SystemExit(f"No files found for {args.source or 'config dataset.video'} (pattern={args.pattern})")

    runner = Path(__file__).parent / "run_track.py"
    for f in files:
        # pass per-run overrides to set dataset.video in the config (run_track reads config.dataset.video)
        run_overrides = list(args.overrides)
        if args.visualize:
            run_overrides.append("dataset.visualize=true")
        # use plain override (no leading '+') because dataset.video already exists in the config
        cmd = [sys.executable, str(runner), "--config-name", args.config_name, f'dataset.video={str(f)}'] + run_overrides
        print(f"Running: {' '.join(cmd)}")
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"run_track failed for {f} (exit {res.returncode})")


if __name__ == "__main__":
    main()