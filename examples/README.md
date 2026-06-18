# Traffic Node Video Dataset 2.0 sample code

These scripts demonstrate the minimal workflow used on the GitHub Pages documentation page.

The examples use the small files in `sample_data/`. After the full dataset is uploaded to Google Drive, replace the paths with the downloaded real files.

## Run

```bash
pip install -r requirements.txt
python examples/load_dataset.py
python examples/plot_trajectories.py
python examples/apply_homography.py
python examples/generate_summary_stats.py
```

## Data expected by the examples

- `sample_data/dataset_manifest_sample.csv`
- `sample_data/sample_trajectories.csv`
- `sample_data/sample_homography.json`
