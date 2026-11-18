"""Clip a tracker result file (MOT format) to the sequence's GT eval range.

VNTraffic keeps frames 1..500 (video has 510 frames); AICC22-Custom keeps 3..200
(frames 1-2 have no GT, predictions there would count as false positives).

Usage:
    python tools/filter_preds.py --seq VNTraffic --inp raw.txt --out data/VNTraffic.txt
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emo_ocsort import paths  # noqa: E402

DEFAULT_META = str(paths.sequences_json())


def eval_range(seq, meta_path=DEFAULT_META):
    return json.loads(Path(meta_path).read_text())["sequences"][seq]["eval_range"]


def filter_file(inp, out, seq, meta_path=DEFAULT_META):
    lo, hi = eval_range(seq, meta_path)
    kept = removed = 0
    lines = []
    for line in Path(inp).read_text().splitlines():
        if not line.strip():
            continue
        fr = int(float(line.split(",")[0]))
        if lo <= fr <= hi:
            lines.append(line)
            kept += 1
        else:
            removed += 1
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + ("\n" if lines else ""), newline="\n")
    return kept, removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default=DEFAULT_META)
    a = ap.parse_args()
    kept, removed = filter_file(a.inp, a.out, a.seq, a.meta)
    print(f"{a.seq}: kept {kept} rows, removed {removed} rows outside eval range {eval_range(a.seq, a.meta)}")


if __name__ == "__main__":
    main()
