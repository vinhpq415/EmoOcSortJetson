"""Resolve data / weights / third-party locations.

Priority for every path: environment variable > default under EMO_BASE (default: ~/emo, the layout
of docs/JETSON.md).

    EMO_BASE           base folder (default ~/emo)
    EMO_DATA_ROOT      folder holding sequences.json        (default $EMO_BASE/Vehicle_Tracking_MOT)
    EMO_WEIGHTS        yolov5x.pt, one .engine, or a folder of yolov5x_b<N>.engine
                                                            (default $EMO_BASE/yolov5x.pt)
    EMO_YOLOV5_DIR     ultralytics/yolov5 clone @ af08ba15  (default $EMO_BASE/yolov5)
    EMO_TRACKEVAL_DIR  JonathonLuiten/TrackEval @ 12c8791   (default <repo>/third_party/TrackEval)
    EMO_OUTPUT_ROOT    where results are written            (default <repo>/outputs)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repository root

YOLOV5_COMMIT = "af08ba156fd58ed5b0fe7e4e58a22c72c3cdd0ce"
TRACKEVAL_COMMIT = "12c8791"


def _env(name, default):
    v = os.environ.get(name)
    return Path(v).expanduser() if v else Path(default).expanduser()


def base():
    return _env("EMO_BASE", Path.home() / "emo")


def data_root():
    return _env("EMO_DATA_ROOT", base() / "Vehicle_Tracking_MOT")


def weights():
    return _env("EMO_WEIGHTS", base() / "yolov5x.pt")


def yolov5_dir():
    return _env("EMO_YOLOV5_DIR", base() / "yolov5")


def trackeval_dir():
    return _env("EMO_TRACKEVAL_DIR", ROOT / "third_party" / "TrackEval")


def output_root():
    return _env("EMO_OUTPUT_ROOT", ROOT / "outputs")


def sequences_json():
    return data_root() / "sequences.json"


def gt_folder():
    return data_root() / "trackeval" / "gt" / "mot_challenge"


def video_path(seq):
    return data_root() / "videos" / f"{seq}.mp4"


def summary():
    items = {
        "code_root": ROOT,
        "data_root": data_root(),
        "weights": weights(),
        "yolov5_dir": yolov5_dir(),
        "trackeval_dir": trackeval_dir(),
        "output_root": output_root(),
    }
    return {k: str(v) for k, v in items.items()}
