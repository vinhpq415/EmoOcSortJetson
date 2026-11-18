"""Build fixed-batch FP16 TensorRT engines of yolov5x for batch sizes 1..N (Jetson, Phase 5).

YOLOv5's exporter refuses --half together with --dynamic, so one engine per batch size is built; the
detector (emo_ocsort/detector.py) loads the directory and picks the engine matching the batch.
An engine only runs on the TensorRT version and GPU it was built with: build it ON the Jetson.

    python tools/export_engines.py --out ~/emo/engines             # default batch sizes 1 2 3 4
    export EMO_WEIGHTS=~/emo/engines                                # then every tool uses TensorRT
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emo_ocsort import paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="yolov5x.pt (default: paths.weights())")
    ap.add_argument("--out", required=True, help="output directory for yolov5x_b<N>.engine")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--imgsz", type=int, nargs=2, default=[384, 640], help="h w (must match the detector input_hw)")
    ap.add_argument("--workspace", type=int, default=4, help="TensorRT workspace in GB")
    a = ap.parse_args()

    weights = Path(a.weights or paths.weights())
    if weights.suffix != ".pt":
        raise SystemExit(f"--weights must be the PyTorch .pt file, got {weights}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    work.mkdir(exist_ok=True)
    src = work / weights.name  # export writes next to the weights: keep the original folder clean
    shutil.copy2(weights, src)
    exporter = paths.yolov5_dir() / "export.py"
    try:
        for b in a.batches:
            t = time.time()
            cmd = [sys.executable, str(exporter), "--weights", str(src), "--include", "engine", "--half",
                   "--device", "0", "--imgsz", *map(str, a.imgsz), "--batch-size", str(b),
                   "--workspace", str(a.workspace)]
            print(f"[export] batch {b}: {' '.join(cmd)}", flush=True)
            r = subprocess.run(cmd, cwd=str(paths.yolov5_dir()))
            built = src.with_suffix(".engine")
            if r.returncode != 0 or not built.exists():
                raise SystemExit(f"engine export failed for batch {b} (is the tensorrt Python package importable?)")
            dst = out / f"{weights.stem}_b{b}.engine"
            shutil.move(str(built), dst)
            print(f"[export] -> {dst} ({dst.stat().st_size / 2**20:.0f} MB, {time.time() - t:.0f} s)", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)  # temporary copy of the weights + ONNX
    print(f"done. Use: EMO_WEIGHTS={out}")


if __name__ == "__main__":
    main()
