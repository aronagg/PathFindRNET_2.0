import json
from pathlib import Path

from traffic.io.legacy_io import load_legacy_json


def _sample_entity():
    return {
        "id": 3,
        "history_X": [0.0, 0.1],
        "history_Y": [0.0, 0.1],
        "detections": [
            {
                "label": "car",
                "confidence": 0.62,
                "X": 0.4875,
                "Y": 0.2597,
                "Width": 0.0861,
                "Height": 0.1389,
                "frameID": 9.0,
            },
            {
                "label": "car",
                "confidence": 0.63,
                "X": 0.4882,
                "Y": 0.2590,
                "Width": 0.0875,
                "Height": 0.1375,
                "frameID": 10.0,
            },
        ],
    }


def test_load_legacy_entities_array(tmp_path: Path):
    ent = _sample_entity()
    path = tmp_path / "entities.json"
    path.write_text(json.dumps([ent]))

    df = load_legacy_json(path, class_map={"car": 2})
    assert list(df.columns) == ["frame", "track_id", "cls", "conf", "cx", "cy", "w", "h"]
    assert len(df) == 2
    assert set(df["frame"]) == {9, 10}
    assert set(df["track_id"]) == {3}
    assert set(df["cls"]) == {2}
    # spot check values
    row = df.sort_values("frame").iloc[0]
    assert abs(row.cx - 0.4875) < 1e-6
    assert abs(row.w - 0.0861) < 1e-6


def test_load_legacy_entities_ndjson(tmp_path: Path):
    ent = _sample_entity()
    path = tmp_path / "entities.ndjson"
    path.write_text("\n".join(json.dumps(x) for x in [ent, ent]))

    df = load_legacy_json(path)
    # Without class_map, cls should be -1
    assert set(df["cls"]) == {-1}
    assert len(df) == 4  # two entities, each 2 detections


def test_history_fallback(tmp_path: Path):
    ent = {"id": 7, "history_X": [0.2, 0.3, 0.4], "history_Y": [0.5, 0.6, 0.7]}
    path = tmp_path / "hist.json"
    path.write_text(json.dumps([ent]))

    df = load_legacy_json(path)
    assert len(df) == 3
    assert set(df["frame"]) == {0, 1, 2}
    assert set(df["track_id"]) == {7}
