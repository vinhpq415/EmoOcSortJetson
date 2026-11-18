# emo-ocsort-jetson

Bỏ qua detection có chủ đích kiểu **EMO** ([Ganesh et al., arXiv:2309.02666](https://arxiv.org/abs/2309.02666))
cho **YOLOv5x (weights COCO gốc) + OC-SORT**, chạy **nhiều luồng video trên một GPU**. Đích triển khai:
**Jetson Orin AGX** (TensorRT FP16).

> **Trạng thái:** đã đo trên GPU T4 (PyTorch FP16 + CUDA Graph). **Chưa chạy trên Jetson thật**; phần TensorRT
> (`tools/export_engines.py`, nhánh `.engine` của `emo_ocsort/detector.py`) chưa được kiểm tra trên thiết bị.

## Kết quả tham chiếu (GPU T4, 4 luồng VNTraffic 1080p 30 FPS = 120 frame/s)

| Cấu hình | Frame rơi | Độ trễ TB / p95 | HOTA |
|---|---|---|---|
| 1 luồng, detect mọi frame (đối chứng) | 1.2% | 24 / 26 ms | 73.57 |
| 4 luồng, detect mọi frame | 79.2% | 150 / 228 ms | 15.36 |
| **4 luồng, B3 k = 3 (cấu hình khuyến nghị)** | **0.5%** | **17 / 53 ms** | **68.18** |

Cấu hình khuyến nghị:
- skip theo chu kỳ **k = 3** (detect 1 trong 4 frame) + dự đoán Kalman;
- OC-SORT xử lý frame skip kiểu **V1** với **ORU nhanh** (`--skip_oru False`);
- vòng lặp **bất đồng bộ** (GPU và CPU chạy song song);
- dịch chuyển tracker qua frame rơi **bằng một bước Kalman**;
- forward bằng **CUDA Graph** (PyTorch) hoặc **TensorRT**.

Quyết định skip theo độ tương đồng ảnh (NCC) của paper không được khuyến nghị khi chạy real-time: trên T4 nó
tốn khoảng 5 ms CPU mỗi frame và cho HOTA khoảng 33.

## Chuẩn bị

Chi tiết từng bước (JetPack, PyTorch của NVIDIA, TensorRT, chế độ năng lượng) ở **[docs/JETSON.md](docs/JETSON.md)**.

Cấu trúc thư mục mặc định (`EMO_BASE=~/emo`):

```
~/emo/
  emo-ocsort-jetson/                      # repo này
  emo-ocsort-jetson/third_party/TrackEval # git clone https://github.com/JonathonLuiten/TrackEval && git checkout 12c8791
  yolov5/                                 # git clone https://github.com/ultralytics/yolov5 && git checkout af08ba156fd58ed5b0fe7e4e58a22c72c3cdd0ce
  yolov5x.pt                              # https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt
  Vehicle_Tracking_MOT/                   # dữ liệu (không nằm trong repo)
```

`yolov5x.pt` dùng khi đo có SHA-256 `9f27a794fa0308e2606f90565164571d6f0a0ba18a3ce2e5e5d323b71c157859`.

**Dữ liệu** (`Vehicle_Tracking_MOT/`, không phân phối kèm repo) có cấu trúc:

```
videos/<seq>.mp4
sequences.json                                     # fps, kích thước, seq_length, eval_range của từng video
trackeval/gt/mot_challenge/VEHICLE-test/<seq>/gt/gt.txt   # frame,id,x,y,w,h,1,1,1 (frame đếm từ 1)
trackeval/gt/mot_challenge/VEHICLE-test/<seq>/seqinfo.ini
trackeval/gt/mot_challenge/seqmaps/VEHICLE-test.txt
```

Mọi đường dẫn đều ghi đè được bằng biến môi trường: `EMO_BASE`, `EMO_DATA_ROOT`, `EMO_WEIGHTS`
(file `.pt`, một file `.engine`, hoặc thư mục chứa `yolov5x_b<N>.engine`), `EMO_YOLOV5_DIR`,
`EMO_TRACKEVAL_DIR`, `EMO_OUTPUT_ROOT` (xem `emo_ocsort/paths.py`).

Gói Python (**không** cài torch từ PyPI trên Jetson; dùng wheel của NVIDIA):

```
pip install -r requirements-jetson.txt
pip install ultralytics==8.4.194 --no-deps
```

## Chạy

Toàn bộ quy trình đo trên Jetson: MAXN, `jetson_clocks`, ghi `tegrastats`, kiểm tra môi trường, build engine
TensorRT, chạy 1 luồng, chạy 4 luồng (quét k, offline), so sánh TensorRT với PyTorch:

```
cd ~/emo/emo-ocsort-jetson
bash tools/jetson_run.sh
```

Các lệnh riêng lẻ:

```
python tools/env_check.py                                   # kiểm tra môi trường (phải in ENV CHECK PASSED)
python tools/export_engines.py --out ~/emo/engines          # engine FP16 batch 1..4 (build trên chính thiết bị)
export EMO_WEIGHTS=~/emo/engines                            # từ đây mọi lệnh dùng TensorRT
python tools/run_single.py --exp B1 --save_video            # 1 luồng, detect mọi frame, 2 video
python tools/run_multi.py --exp B3k3 --policy periodic_est --k 3 --skip_oru False   # 4 luồng real-time
python tools/run_phase4.py rt1 ksweep off7 --phase jetson   # ma trận thí nghiệm
python tools/summarize_multi.py --phase jetson              # bảng tổng hợp -> outputs/jetson/summary.md
```

**Chọn lại k trên thiết bị:** chọn k nhỏ nhất sao cho 4 luồng real-time rơi ≤ 1–2% frame và độ trễ p95
≤ khoảng 100 ms. Trên T4, đó là k = 3.

## Cấu trúc code

```
emo_ocsort/
  paths.py         đường dẫn (biến môi trường > mặc định dưới EMO_BASE)
  detector.py      YOLOv5x: PyTorch (CUDA Graph theo batch size) hoặc thư mục engine TensorRT; letterbox 384x640
  tracker.py       bọc OC-SORT (det_thresh 0.3, skip_mode V1, skip_oru)
  emo_policy.py    chính sách skip: none / reuse / periodic_est / ncc, AFD
  pipeline.py      vòng lặp 1 luồng
  multistream.py   N luồng, real-time / offline, đồng bộ / bất đồng bộ, dịch chuyển qua frame rơi
  ocsort_tracker/  OC-SORT (MIT, noahcao/OC_SORT @ 8462e7e); các chỗ sửa cho EMO đánh dấu "# EMO:"
  dets_cache.py, video.py, vis.py
tools/
  env_check.py, export_engines.py, jetson_run.sh
  run_single.py, run_multi.py, run_phase4.py, summarize_multi.py, gpu_gap_bench.py
  run_trackeval.py, filter_preds.py
docs/JETSON.md
```

## Ghi công

- EMO: S. V. Ganesh, Y. Wu, G. Liu, R. Kompella, L. Liu, *Fast and Resource-Efficient Object Tracking on Edge
  Devices: A Measurement Study*, arXiv:2309.02666; repo [git-disl/EMO](https://github.com/git-disl/EMO).
  Logic quyết định skip được viết lại cho OC-SORT; không dùng code FairMOT.
- OC-SORT: [noahcao/OC_SORT](https://github.com/noahcao/OC_SORT) (MIT), có trong `emo_ocsort/ocsort_tracker/`
  kèm LICENSE gốc.
- YOLOv5: [ultralytics/yolov5](https://github.com/ultralytics/yolov5) (AGPL-3.0), không kèm trong repo, được
  gọi khi chạy.
- TrackEval: [JonathonLuiten/TrackEval](https://github.com/JonathonLuiten/TrackEval) (MIT), không kèm trong repo.

## Giấy phép

Code của repo này phát hành theo **GNU AGPL-3.0** (xem [LICENSE](LICENSE)), cùng giấy phép với YOLOv5 mà nó
chạy cùng. Thư mục `emo_ocsort/ocsort_tracker/` giữ giấy phép MIT gốc của OC-SORT (file LICENSE trong thư mục
đó).
