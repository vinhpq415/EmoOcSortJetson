"""Multi-stream experiment matrix: 4 staggered copies of VNTraffic on one GPU.

Defaults of tools/run_multi.py: async GPU/CPU overlap, closed-form advance over dropped frames, dropped
frames not counted toward k, CUDA-graph forward (PyTorch weights; TensorRT engines need none) and a 0.5 ms
GIL switch interval.

Groups
  rt1     real-time, 1 stream, B1 (reference without overload)
  ksweep  real-time, 4 streams: B1, B3 k2..k6 (fast ORU), NCC 0.75 k4/k6 (fast ORU)
  o7      real-time B3 k3 / k6 with CUDA graph and/or switch interval off (PyTorch weights only)
  off7    offline (no drops): B1, B3 k3, B3 k6 (default / without O7 / synchronous)

    python tools/run_phase4.py rt1 ksweep off7 --phase jetson_trt
    python tools/summarize_multi.py --phase jetson_trt
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emo_ocsort import paths  # noqa: E402

B3 = lambda k: ["--policy", "periodic_est", "--k", str(k)]  # noqa: E731
NCC = lambda th, k: ["--policy", "ncc", "--k", str(k), "--sim_thresh", str(th)]  # noqa: E731
FAST = ["--skip_oru", "False"]
OFF = ["--mode", "offline"]
NO_O7 = ["--cuda_graph", "0", "--switch_interval", "0"]


def rt1():
    yield "RT1_B1", ["--streams", "1", "--offsets", "0"]


def ksweep():
    yield "RT_B1", []
    for k in (2, 3, 4, 5, 6):
        yield f"RT_B3_k{k}_fast", B3(k) + FAST
    yield "RT_NCC0.75_k4_fast", NCC(0.75, 4) + FAST
    yield "RT_NCC0.75_k6_fast", NCC(0.75, 6) + FAST


def o7():
    for k in (3, 6):
        yield f"O7_B3_k{k}_fast_none", B3(k) + FAST + NO_O7
        yield f"O7_B3_k{k}_fast_graphonly", B3(k) + FAST + ["--switch_interval", "0"]
        yield f"O7_B3_k{k}_fast_switchonly", B3(k) + FAST + ["--cuda_graph", "0"]


def off7():
    yield "OFF_B1", OFF
    yield "OFF_B3_k3_fast", OFF + B3(3) + FAST
    yield "OFF_B3_k6_fast", OFF + B3(6) + FAST
    yield "OFF_B3_k6_fast_noO7", OFF + B3(6) + FAST + NO_O7
    yield "OFF_B3_k6_fast_sync", OFF + B3(6) + FAST + ["--async_gpu", "0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+", choices=["rt1", "ksweep", "o7", "off7"])
    ap.add_argument("--phase", default="multi", help="output subfolder under EMO_OUTPUT_ROOT")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    gens = {"rt1": rt1, "ksweep": ksweep, "o7": o7, "off7": off7}
    for g in a.groups:
        for exp, args in gens[g]():
            if (paths.output_root() / a.phase / exp / "metrics.json").exists() and not a.force:
                print(f"[skip] {exp} (done)")
                continue
            print(f"[run ] {exp}: {' '.join(args)}", flush=True)
            r = subprocess.run([sys.executable, str(ROOT / "tools" / "run_multi.py"), "--phase", a.phase,
                                "--exp", exp, *args], capture_output=True, text=True)
            keep = [l for l in r.stdout.splitlines() if l.startswith(f"[{exp}]") and "stream {" not in l]
            print("\n".join(keep) if r.returncode == 0 else r.stdout[-3000:] + r.stderr[-3000:], flush=True)
            if r.returncode != 0:
                raise SystemExit(f"{exp} failed")


if __name__ == "__main__":
    main()
