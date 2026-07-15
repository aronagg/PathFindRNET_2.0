# Publication Result Generation

This folder contains annotation-free publication tables and figures for the Traffic Node Video Dataset 2.0 paper.

## Reproduce

```bash
python scripts/generate_publication_results.py --new-root "PATH_TO_TRAFFIC_NODE_VIDEO_DATASET_2_0" --out reports/publication_results
python scripts/generate_publication_results.py --new-root "PATH_TO_TRAFFIC_NODE_VIDEO_DATASET_2_0" --old-root "PATH_TO_PREVIOUS_TRAFFIC_NODE_DATASET" --out reports/publication_results
```

Use the repository virtual environment if the system Python does not include pandas, pyarrow, matplotlib, and joblib:

```powershell
.\.venv\Scripts\python.exe scripts\generate_publication_results.py --new-root "TNVD2_UPLOAD_PACKAGE" --old-root "old_data" --out reports\publication_results
```

## Scientific Note

The script does not use manual annotations and does not report ground-truth detector accuracy. It reports annotation-free indicators of dataset scale, trajectory continuity, gap behavior, class consistency, smoothness, cluster/outlier behavior, and homography metadata availability.
