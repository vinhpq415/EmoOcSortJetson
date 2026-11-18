"""EMO detection-skipping policies (Ganesh et al., arXiv:2309.02666; repo git-disl/EMO src/track.py).

Policies (name -> paper / repo counterpart)
  none          B1: detect every frame (no skipping)
  reuse         B2: periodic skip, reuse the previous boxes unchanged (track_periodic.py, "Alt skip")
  periodic_est  B3: periodic skip + Kalman estimation ("Alt skip + estimation"; track.py with
                    --similarity_computation no)
  ncc           E1: context-aware skip with NCC similarity + Kalman estimation (+ AFD)

Decision for `ncc` (Algorithm 1, lines 3-7): for every track output on the previous frame, crop the
previous frame at the track's previous box and the current frame at its Kalman-predicted box, resize
both to 128x128 grayscale, compute the normalized cross-correlation (cv2.TM_CCOEFF_NORMED), average
over tracks. Skip when avg >= sim_thresh and fewer than k consecutive frames were skipped.

Differences from the EMO repo, on purpose:
  - no `w/h > 1.6` filter (it drops most vehicle boxes; it is for pedestrians)
  - boxes are clipped to the image instead of PIL's black padding; tracks whose clipped crop is
    smaller than min_box_area are left out of the average
  - the similarity is not computed when the k-consecutive-skips limit already forces a detection
  - cv2 INTER_LINEAR resize instead of PIL's default resampling
"""
import time

import cv2
import numpy as np

POLICIES = ("none", "reuse", "periodic_est", "ncc")
CROP = (128, 128)


def afd_interval(k, fps, enabled):
    """Adaptive forced detection (repo: detect_frame_interval / 4 when FPS < 15)."""
    return int(k / 4) if enabled and fps < 15 else k


def _crop(gray, box):
    h, w = gray.shape
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return None
    return gray[y1:y2, x1:x2]


def ncc_similarity_reference(prev_gray, cur_gray, pairs, min_box_area=100.0):
    """Reference implementation (EMO repo): one cv2.matchTemplate per track. Kept for equivalence tests."""
    total, n = 0.0, 0
    for prev_box, pred_box in pairs:
        a = _crop(prev_gray, prev_box)
        b = _crop(cur_gray, pred_box)
        if a is None or b is None or a.size <= min_box_area or b.size <= min_box_area:
            continue
        a = cv2.resize(a, CROP, interpolation=cv2.INTER_LINEAR)
        b = cv2.resize(b, CROP, interpolation=cv2.INTER_LINEAR)
        r = float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0])
        if np.isfinite(r):  # flat crops give NaN
            total += r
            n += 1
    return (total / n if n else 0.0), n


def ncc_similarity(prev_gray, cur_gray, pairs, min_box_area=100.0):
    """Average NCC over (prev_box, predicted_box) pairs. Returns (avg, n_used); avg = 0 if no pair.

    Same quantity as cv2.matchTemplate(TM_CCOEFF_NORMED) on two equal-size crops (= Pearson correlation
    of the pixels), computed for all tracks at once with numpy. matchTemplate runs a general sliding
    template search (~0.2 ms per call); here the whole frame costs one vectorized pass (Phase 4, O1).
    """
    A, B = [], []
    for prev_box, pred_box in pairs:
        a = _crop(prev_gray, prev_box)
        b = _crop(cur_gray, pred_box)
        if a is None or b is None or a.size <= min_box_area or b.size <= min_box_area:
            continue
        A.append(cv2.resize(a, CROP, interpolation=cv2.INTER_LINEAR))
        B.append(cv2.resize(b, CROP, interpolation=cv2.INTER_LINEAR))
    if not A:
        return 0.0, 0
    # float32 like matchTemplate; |difference| to matchTemplate ~1e-6 (3.4x faster for ~27 tracks)
    a = np.asarray(A, np.float32).reshape(len(A), -1)
    b = np.asarray(B, np.float32).reshape(len(B), -1)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    den = np.sqrt(np.einsum("ij,ij->i", a, a).astype(np.float64) * np.einsum("ij,ij->i", b, b))
    ok = den > 0  # flat crops: matchTemplate gives NaN and they are left out
    if not ok.any():
        return 0.0, 0
    r = np.einsum("ij,ij->i", a[ok], b[ok]) / den[ok]
    return float(r.mean()), int(ok.sum())


class SkipPolicy:
    def __init__(self, name="none", k=4, sim_thresh=0.75, afd=True, fps=30.0, min_box_area=100.0,
                 drops_count=False):
        if name not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        self.name, self.sim_thresh, self.min_box_area = name, sim_thresh, min_box_area
        self.drops_count = drops_count
        self.k = afd_interval(k, fps, afd) if name == "ncc" else k
        self.consecutive = 0
        self.prev_gray = None
        self.first = True  # the first processed frame is always detected

    def note_dropped(self, n):
        """Multi-stream real-time mode: n frames were dropped (never processed) before the current one.

        drops_count=False (default): dropped frames do NOT count toward the k-consecutive-skips limit.
        Counting them (drops_count=True, first Phase 4 run) forces a detection after every burst of drops,
        i.e. the expensive operation exactly when the system is overloaded -> more drops (a feedback loop
        seen on T4 and locally). Not counting them lets the policy skip k cheap processed frames after each
        detection, so the load regulates itself."""
        if self.drops_count:
            self.consecutive += n

    def decide(self, frame, fid, tracker):
        """Return (skip: bool, info: dict) for the current frame. Must be followed by commit()."""
        info = {"sim": None, "n_sim": 0, "decision_ms": 0.0}
        if self.name == "none" or self.first or self.k <= 0:
            return False, info
        if self.consecutive >= self.k:
            return False, info  # forced detection
        if self.name in ("reuse", "periodic_est"):
            return True, info
        t = time.perf_counter()
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg, n = ncc_similarity(self.prev_gray, cur_gray, tracker.peek_boxes(), self.min_box_area)
        info.update(sim=avg, n_sim=n, decision_ms=1e3 * (time.perf_counter() - t))
        self._cur_gray = cur_gray
        return avg >= self.sim_thresh, info

    def commit(self, frame, skipped):
        """Record the outcome; keep the grayscale frame for the next decision (ncc only).
        Returns the extra decision-related time in ms (grayscale conversion when decide() did not do it)."""
        self.consecutive = self.consecutive + 1 if skipped else 0
        self.first = False
        if self.name != "ncc":
            return 0.0
        t = time.perf_counter()
        g = getattr(self, "_cur_gray", None)
        self.prev_gray = g if g is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._cur_gray = None
        return 1e3 * (time.perf_counter() - t)
