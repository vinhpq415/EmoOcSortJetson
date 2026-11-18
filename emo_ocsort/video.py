"""Frame source. Frame ids are 1-based to match the GT convention (gt_frame = video_index + 1)."""
import time

import cv2


class VideoReader:
    def __init__(self, path, max_frames=None):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {self.path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.max_frames = max_frames

    def __iter__(self):
        """Yields (frame_id, frame_bgr, decode_ms)."""
        fid = 0
        while self.max_frames is None or fid < self.max_frames:
            t = time.perf_counter()
            ok, frame = self.cap.read()
            dt = 1e3 * (time.perf_counter() - t)
            if not ok:
                break
            fid += 1
            yield fid, frame, dt
        self.cap.release()
