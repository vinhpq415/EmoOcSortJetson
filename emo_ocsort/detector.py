"""YOLOv5 (official COCO weights) detector wrapper used by every phase.

- Fixed letterbox input (default 384x640 = imgsz 640 for 16:9 video), so frames from any
  resolution can share one batch (multi-stream) and the TensorRT engine can use a fixed shape.
- Returns, per frame, an (N, 6) float32 array [x1, y1, x2, y2, score, cls] in ORIGINAL image
  pixels; score = objectness * class prob (what OC-SORT expects as the 5th column).
- Uses ultralytics/yolov5 code (DetectMultiBackend) so a .engine file can replace the .pt later.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import paths

COCO_VEHICLES = (2, 3, 5, 7)  # car, motorcycle, bus, truck


def _import_yolov5():
    yd = str(paths.yolov5_dir())
    if yd not in sys.path:
        sys.path.insert(0, yd)
    from models.common import DetectMultiBackend
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_boxes
    return DetectMultiBackend, letterbox, non_max_suppression, scale_boxes


def auto_fp16(device):
    """FP16 only where it is fast: Tensor-Core GPUs (T4, Orin...). GTX 16xx has none and runs
    FP16 ~5x slower than FP32 (measured locally: 351 vs 73 ms/frame for yolov5x)."""
    if device.type != "cuda":
        return False
    name = torch.cuda.get_device_name(device)
    return torch.cuda.get_device_capability(device) >= (7, 0) and "GTX 16" not in name


class YoloV5Detector:
    def __init__(self, weights=None, device=None, fp16=None, input_hw=(384, 640),
                 conf_thres=0.1, iou_thres=0.45, classes=COCO_VEHICLES, agnostic_nms=True, max_det=300,
                 cuda_graph=False):
        DetectMultiBackend, self._letterbox, self._nms, self._scale_boxes = _import_yolov5()
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.fp16 = auto_fp16(self.device) if fp16 is None else bool(fp16)
        self.weights = str(weights or paths.weights())
        # TensorRT (Jetson, Phase 5): a directory of fixed-batch FP16 engines yolov5x_b<N>.engine made by
        # tools/export_engines.py (YOLOv5's exporter does not allow --half with --dynamic), or one .engine.
        self.engines = {}
        wp = Path(self.weights)
        if wp.is_dir() or wp.suffix == ".engine":
            files = sorted(wp.glob("*_b*.engine")) if wp.is_dir() else [wp]
            if not files:
                raise FileNotFoundError(f"no *_b<N>.engine files in {wp}")
            coco = str(paths.yolov5_dir() / "data" / "coco128.yaml")  # engines carry no class names
            for f in files:
                m = DetectMultiBackend(str(f), device=self.device, fp16=True, data=coco)
                self.engines[int(m.bindings["images"].shape[0])] = m
            self.model = self.engines[min(self.engines)]
            self.fp16 = bool(self.model.fp16)
            cuda_graph = False  # TensorRT runs a whole forward with one execute call already
        else:
            self.model = DetectMultiBackend(self.weights, device=self.device, fp16=self.fp16)
        self.backend = "tensorrt" if self.engines else "pytorch"
        self.names = self.model.names
        self.stride = int(self.model.stride)
        self.input_hw = tuple(input_hw)
        if any(s % self.stride for s in self.input_hw):
            raise ValueError(f"input_hw {self.input_hw} must be a multiple of stride {self.stride}")
        self.conf_thres, self.iou_thres = conf_thres, iou_thres
        self.classes = list(classes) if classes is not None else None
        self.agnostic_nms, self.max_det = agnostic_nms, max_det
        self.last_timing = {}
        # Phase 4 O7: CUDA graphs, one per batch size, captured lazily (capture_graphs / warmup)
        self.cuda_graph = bool(cuda_graph) and self.device.type == "cuda"
        self._graphs = {}  # batch -> (graph, static_in, static_out)

    def capture_graphs(self, batch_sizes):
        """Record the network forward as a CUDA graph per batch size. A replay is ONE launch instead of
        hundreds of Python-side kernel launches, so a detection thread barely needs the GIL (O7)."""
        if not self.cuda_graph:
            return
        net = self.model.model  # the YOLOv5 DetectionModel inside DetectMultiBackend
        dtype = torch.float16 if self.fp16 else torch.float32
        for b in batch_sizes:
            if b in self._graphs:
                continue
            static_in = torch.zeros((b, 3, *self.input_hw), device=self.device, dtype=dtype)
            side = torch.cuda.Stream(self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.no_grad(), torch.cuda.stream(side):
                for _ in range(3):  # warm-up on a side stream (also builds the Detect grids)
                    net(static_in)
            torch.cuda.current_stream(self.device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(graph):
                out = net(static_in)
            static_out = out[0] if isinstance(out, (list, tuple)) else out
            self._graphs[b] = (graph, static_in, static_out)
        torch.cuda.synchronize(self.device)

    def _forward(self, x):
        if self.engines:
            m = self.engines.get(x.shape[0])
            if m is None:
                raise ValueError(f"no TensorRT engine for batch {x.shape[0]} (have {sorted(self.engines)})")
            pred = m(x)
            return pred[0] if isinstance(pred, (list, tuple)) else pred
        g = self._graphs.get(x.shape[0]) if self.cuda_graph else None
        if g is None:
            pred = self.model(x)
            return pred[0] if isinstance(pred, (list, tuple)) else pred
        graph, static_in, static_out = g
        static_in.copy_(x)
        graph.replay()
        return static_out  # read by NMS before the next replay of this graph (same thread)

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def preprocess(self, frames):
        batch = np.stack([self._letterbox(f, self.input_hw, stride=self.stride, auto=False)[0] for f in frames])
        x = torch.from_numpy(np.ascontiguousarray(batch[..., ::-1].transpose(0, 3, 1, 2))).to(self.device)
        x = x.half() if self.fp16 else x.float()
        return x / 255.0

    @torch.no_grad()
    def __call__(self, frames):
        """frames: list of BGR uint8 images (any sizes). Returns list of (N, 6) arrays."""
        if len(frames) == 0:
            return []
        t0 = time.perf_counter()
        x = self.preprocess(frames)
        self._sync()
        t1 = time.perf_counter()
        pred = self._forward(x)
        self._sync()
        t2 = time.perf_counter()
        dets = self._nms(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                         agnostic=self.agnostic_nms, max_det=self.max_det)
        out = []
        for d, f in zip(dets, frames):
            if len(d):
                d[:, :4] = self._scale_boxes(self.input_hw, d[:, :4], f.shape)
            out.append(d.float().cpu().numpy())
        t3 = time.perf_counter()
        self.last_timing = {"pre_ms": 1e3 * (t1 - t0), "infer_ms": 1e3 * (t2 - t1),
                            "post_ms": 1e3 * (t3 - t2), "batch": len(frames)}
        return out

    def warmup(self, batch=1, iters=3):
        self.capture_graphs([batch])
        dummy = [np.zeros((*self.input_hw, 3), np.uint8)] * batch
        for _ in range(iters):
            self(dummy)
