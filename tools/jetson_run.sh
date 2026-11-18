#!/usr/bin/env bash
# Phase 5 - full measurement sequence on a Jetson Orin AGX (see docs/JETSON.md first).
# Run from the repository folder:  bash tools/jetson_run.sh
# Override any path with an environment variable, e.g.  PT=/data/yolov5x.pt bash tools/jetson_run.sh
set -euo pipefail

BASE=${BASE:-$HOME/emo}
export EMO_DATA_ROOT=${EMO_DATA_ROOT:-$BASE/Vehicle_Tracking_MOT}
export EMO_YOLOV5_DIR=${EMO_YOLOV5_DIR:-$BASE/yolov5}            # ultralytics/yolov5 @ af08ba15
export EMO_TRACKEVAL_DIR=${EMO_TRACKEVAL_DIR:-$PWD/third_party/TrackEval}   # @ 12c8791
export EMO_OUTPUT_ROOT=${EMO_OUTPUT_ROOT:-$PWD/outputs}
PT=${PT:-$BASE/yolov5x.pt}
ENG=${ENG:-$BASE/engines}
PY=${PY:-python3}
mkdir -p "$EMO_OUTPUT_ROOT"

echo "== 0. power mode MAXN + fixed clocks (needs sudo), tegrastats log"
sudo nvpmodel -m 0 || echo "nvpmodel failed (already MAXN?)"
sudo jetson_clocks || echo "jetson_clocks failed"
sudo nvpmodel -q || true
tegrastats --interval 500 --logfile "$EMO_OUTPUT_ROOT/tegrastats_$(date +%Y%m%d_%H%M%S).log" &
TPID=$!
trap 'kill $TPID 2>/dev/null || true' EXIT

echo "== 1. environment check with the PyTorch weights (FP16 is automatic on Orin)"
EMO_WEIGHTS=$PT $PY tools/env_check.py --precision both

echo "== 2. TensorRT FP16 engines, batch 1..4 (built on THIS device; ~minutes each)"
if [ ! -f "$ENG/yolov5x_b4.engine" ]; then
  $PY tools/export_engines.py --weights "$PT" --out "$ENG"
fi

echo "== 3. environment check with TensorRT"
EMO_WEIGHTS=$ENG $PY tools/env_check.py

echo "== 4. single stream, live detector, both videos (accuracy + per-stage timing)"
EMO_WEIGHTS=$ENG $PY tools/run_single.py --phase jetson --exp B1_live_trt
EMO_WEIGHTS=$PT  $PY tools/run_single.py --phase jetson --exp B1_live_pt

echo "== 5. 4 streams: 1-stream reference, k sweep, offline throughput (TensorRT)"
EMO_WEIGHTS=$ENG $PY tools/run_phase4.py rt1 ksweep off7 --phase jetson_trt
$PY tools/summarize_multi.py --phase jetson_trt

echo "== 6. same with PyTorch FP16 + CUDA graph (comparison)"
EMO_WEIGHTS=$PT $PY tools/run_phase4.py rt1 ksweep --phase jetson_pt
$PY tools/summarize_multi.py --phase jetson_pt

echo "done: results in $EMO_OUTPUT_ROOT/{jetson,jetson_trt,jetson_pt}"
