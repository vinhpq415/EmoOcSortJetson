"""Draw tracks for visual inspection videos."""
import cv2
import numpy as np


def id_color(tid):
    rng = np.random.default_rng(int(tid) * 7919)
    return tuple(int(c) for c in rng.integers(64, 256, 3))


def draw_tracks(frame, tracks, text=""):
    img = frame.copy()
    for x1, y1, x2, y2, tid in tracks:
        c = id_color(tid)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
        cv2.putText(img, str(int(tid)), (int(x1), max(int(y1) - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    if text:
        cv2.putText(img, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return img


class VideoSink:
    def __init__(self, path, fps, width=960):
        self.path, self.fps, self.width, self.w = str(path), fps, width, None

    def write(self, img):
        h, w = img.shape[:2]
        if self.width and w != self.width:
            img = cv2.resize(img, (self.width, int(round(h * self.width / w / 2)) * 2))
        if self.w is None:
            self.w = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (img.shape[1], img.shape[0]))
        self.w.write(img)

    def close(self):
        if self.w is not None:
            self.w.release()
