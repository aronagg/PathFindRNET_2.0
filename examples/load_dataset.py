from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
manifest_path = ROOT / "sample_data" / "dataset_manifest_sample.csv"

manifest = pd.read_csv(manifest_path)
print("Traffic Node Video Dataset 2.0 - sample manifest")
print(manifest)
print()
print(f"Scenes: {manifest['scene_id'].nunique()}")
print(f"Videos: {manifest['video_id'].nunique()}")
