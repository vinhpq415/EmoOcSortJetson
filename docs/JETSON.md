# Chuyển sang Jetson Orin AGX (Phase 5)

Mục tiêu: chạy lại trên Jetson Orin AGX cấu hình đã chốt khi đo trên GPU T4, và đo lại. Cấu hình đó là:
- 4 luồng VNTraffic 1080p 30 FPS;
- B3 k = 3 (skip theo chu kỳ + Kalman, V1, ORU nhanh);
- vòng lặp bất đồng bộ, O3, O7.

**Chưa thử trên Jetson thật.** Phần TensorRT (`tools/export_engines.py`, nhánh `.engine` của `emo_ocsort/detector.py`) chưa chạy được ở local (không có TensorRT). Đây là phần cần kiểm tra đầu tiên.

## 1. Phần cứng và chế độ năng lượng

- [ ] JetPack 6.x (L4T 36.x): CUDA 12.x, cuDNN, TensorRT và Python 3.10 đi kèm. Kiểm tra bằng `cat /etc/nv_tegra_release` và `dpkg -l | grep -i tensorrt`.
- [ ] `sudo nvpmodel -m 0` (MAXN) và `sudo jetson_clocks` (khoá xung nhịp tối đa).
  - Trên T4, việc GPU tự hạ xung nhịp khi rảnh làm forward chậm thêm 15–23%. Khoá xung nhịp loại bỏ hiện tượng này.
- [ ] Theo dõi nhiệt độ và xung nhịp bằng `tegrastats --interval 500 --logfile ...`. Script `tools/jetson_run.sh` tự ghi file này. Chạy dài có thể bị giảm xung nhịp do nhiệt.
- [ ] Bộ nhớ: Orin AGX dùng chung 32 hoặc 64 GB giữa CPU và GPU. Giải mã trước VNTraffic tốn khoảng 3.2 GB, các engine khoảng 4 × 170 MB.

## 2. Môi trường Python (không cài torch từ PyPI)

- [ ] **PyTorch:** dùng wheel NVIDIA build cho đúng phiên bản JetPack (aarch64 + CUDA). Torch từ PyPI không có CUDA cho Jetson. torchvision phải khớp với bản torch đó.
- [ ] **TensorRT Python:** có sẵn trong Python hệ thống của JetPack. Nếu dùng venv, tạo với `python3 -m venv --system-site-packages` để thấy được `tensorrt`.
- [ ] **ultralytics:** repo YOLOv5 @ `af08ba15` cần `ultralytics` (bản đã kiểm tra là 8.4.194).
  - Cài bằng `pip install ultralytics==8.4.194 --no-deps`, rồi cài riêng các phụ thuộc còn thiếu.
  - **Không** để pip kéo torch từ PyPI về, vì sẽ đè lên torch của NVIDIA.
- [ ] Các gói khác:
  - `filterpy`, `scipy`, `pyyaml`, `onnx` (phục vụ bước export).
  - `lap` có thể phải build từ mã nguồn trên aarch64. Nếu không cài được, OC-SORT tự dùng `scipy.optimize.linear_sum_assignment`.
- [ ] OpenCV: bản hệ thống của JetPack có GStreamer (giải mã bằng phần cứng qua `nvv4l2decoder`). Bản `opencv-python` từ pip chỉ giải mã bằng CPU. Để đo, project giải mã trước nên bản nào cũng được; với camera thật thì nên dùng GStreamer.
- [ ] Kiểm tra nhanh: `python3 -c "import torch, tensorrt, cv2, ultralytics; print(torch.cuda.is_available(), tensorrt.__version__)"`.

## 3. Chép sang Jetson

```
$HOME/emo/
  emo-ocsort-jetson/    # repo này (git clone)
  emo-ocsort-jetson/third_party/TrackEval   # git clone + checkout 12c8791
  yolov5/               # git clone ultralytics/yolov5 + checkout af08ba156fd58ed5b0fe7e4e58a22c72c3cdd0ce
  Vehicle_Tracking_MOT/ # dữ liệu đã chuyển đổi (không nằm trong repo)
  yolov5x.pt            # weights COCO v7.0, SHA-256 9f27a794fa0308e2606f90565164571d6f0a0ba18a3ce2e5e5d323b71c157859
```

## 4. Chạy

```
cd $HOME/emo/emo-ocsort-jetson
bash tools/jetson_run.sh          # các bước 0-6, xem bảng dưới
```

| Bước | Nội dung | Đạt khi |
|---|---|---|
| 0 | MAXN, `jetson_clocks`, bắt đầu ghi `tegrastats` | `nvpmodel -q` báo MAXN |
| 1 | `env_check` với `yolov5x.pt` (FP16 tự bật vì Orin có compute capability 8.7) | `ENV CHECK PASSED`; frame 1 của VNTraffic cho khoảng 49 detection |
| 2 | `export_engines.py`: 4 engine FP16 với batch 1, 2, 3, 4, shape 384×640 | Có `engines/yolov5x_b{1..4}.engine` |
| 3 | `env_check` với `EMO_WEIGHTS=engines/` | `ENV CHECK PASSED`; số detection gần bằng bước 1 |
| 4 | 1 luồng, detector thật, cả hai video, TensorRT và PyTorch | HOTA gần với kết quả trên T4 (VNTraffic khoảng 74.6 với `det_thresh = 0.3`) |
| 5 | 4 luồng: đối chứng 1 luồng, quét k = 2–6, NCC, offline (TensorRT) | Xem tiêu chí ở mục 5 |
| 6 | Như bước 5 nhưng dùng PyTorch FP16 + CUDA Graph | Để so với TensorRT |

## 5. Tiêu chí và cách chọn lại k trên Orin

- Chọn **k nhỏ nhất** sao cho, khi chạy real-time 4 luồng:
  - rơi frame **≤ 1–2%**;
  - độ trễ p95 **≤ khoảng 100 ms**.
- Trên T4 kết quả là k = 3: rơi 0.5%, độ trễ trung bình 17 ms, HOTA 68.18. Nếu Orin detect nhanh hơn thì k nhỏ hơn có thể đủ (accuracy cao hơn); nếu chậm hơn thì cần k lớn hơn.
- Ước lượng nhanh trước khi quét: dùng thông lượng offline. Ví dụ trên T4, B3 k = 3 đạt 141 FPS, dư khoảng 17% so với nhu cầu 120 FPS.

## 6. Những gì có thể khác so với T4

- **CPU ARM (Cortex-A78AE)** có thể chậm hơn CPU x86 của máy T4 ở phần Python (OC-SORT, dự đoán Kalman, vòng điều phối). Trên T4, phần CPU đã là nút thắt sau khi GPU được tối ưu. Nếu cần:
  - mỗi luồng chạy tracking trong một process riêng (tránh hoàn toàn GIL);
  - hoặc tăng tốc bộ lọc Kalman (numba, hoặc vector hoá cho mọi track).
- **TensorRT** không cần CUDA Graph: mỗi lần forward chỉ là một lệnh `execute`, nên không có tranh chấp GIL.
- **NCC** không được khuyến nghị: chi phí CPU khoảng 5 ms/frame trên T4, và trên ARM có thể còn lớn hơn.
- **Giải mã video:** khi dùng camera hoặc RTSP thật, phải tính thêm chi phí giải mã. Nên dùng NVDEC qua GStreamer thay vì CPU.
- **INT8** (TensorRT) có thể nhanh hơn nữa nhưng cần calibration và kiểm tra lại accuracy. Đây là bước sau, ngoài phạm vi hiện tại.
