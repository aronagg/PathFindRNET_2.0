.PHONY: setup lint test format run

setup:
	pip install -e .[dev]
	pre-commit install

lint:
	ruff check .
	black --check .
	isort --check-only .
	mypy src

format:
	black .
	isort .

test:
	pytest -q

run:
	python scripts/run_track.py dataset.scene=sample detect=yolo11n tracker=bytetrack source=0
