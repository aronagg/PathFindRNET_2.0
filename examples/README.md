# Traffic Node Video Dataset 2.0 sample code

These scripts demonstrate the sample workflow used on the GitHub Pages documentation page.

The examples use the small files in `sample_data/`. After the full dataset is uploaded to Google Drive, replace the paths with the downloaded real files.

## Basic examples

```bash
pip install -r requirements.txt
python examples/load_dataset.py
python examples/plot_trajectories.py
python examples/apply_homography.py
python examples/generate_summary_stats.py
```

## Extended examples

```bash
python examples/filtering_examples.py
python examples/filling_and_smoothing_examples.py
python examples/homography_extended_examples.py
python examples/isotropic_normalization_examples.py
```

## Covered topics

- Track filtering by length, class, movement and application-specific rules.
- Missing detection filling and short occlusion handling.
- Position, speed, acceleration and bounding rectangle smoothing.
- Homography estimation, image-to-map transformation and reprojection error calculation.
- Scene-level and track-level isotropic coordinate normalization.

## Data expected by the examples

- `sample_data/dataset_manifest_sample.csv`
- `sample_data/sample_trajectories.csv`
- `sample_data/sample_homography.json`
- `sample_data/sample_trajectories_filtering.csv`
- `sample_data/sample_trajectory_with_gaps.csv`
- `sample_data/sample_homography_control_points.csv`
