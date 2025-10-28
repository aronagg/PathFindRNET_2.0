# Traffic Research (Ultralytics + Trajectory Intelligence)

Clean, modular pipeline for traffic trajectory analysis built on **Ultralytics YOLO** tracking + your feature-vector and classifier stack (OPTICS + label consolidation + Re/Ve/Ae vectors + MLP baseline).

## Quickstart

```bash
# 1) Create env
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]

# 2) (Optional) install pre-commit
pre-commit install

# 3) Smoke run (needs a short video or webcam id)
python scripts/run_track.py dataset.scene=sample detect=yolo11n tracker=bytetrack source=0
```
