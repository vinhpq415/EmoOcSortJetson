"""Thin wrapper around the vendored OC-SORT (with the EMO skip additions)."""
import numpy as np

from .ocsort_tracker import OCSort

# OC-SORT defaults (noahcao/OC_SORT) except det_thresh: 0.3 instead of 0.6, fixed in Phase 2 from the
# sweep on both videos (COCO-pretrained detector, not fine-tuned on these scenes).
# skip_mode only matters when a skip policy is active (Phase 3): V0 naive, V1 skipped frames do not
# count toward max_age, V2 they do (see OCSort.skip).
DEFAULT_OCSORT = dict(det_thresh=0.3, max_age=30, min_hits=3, iou_threshold=0.3,
                      delta_t=3, asso_func="iou", inertia=0.2, use_byte=False, skip_mode="V1",
                      skip_oru=True)  # skip_oru=False: Phase 4 fast path (see KalmanBoxTracker.predict_skip)


class OCSortTracker:
    def __init__(self, **cfg):
        self.cfg = {**DEFAULT_OCSORT, **cfg}
        self.t = OCSort(**self.cfg)
        self.last_boxes = {}  # track id (as output, 1-based) -> box [x1, y1, x2, y2] output on the previous frame

    def _remember(self, out):
        self.last_boxes = {int(r[4]): r[:4].copy() for r in out}
        return out

    def update(self, dets, img_hw):
        """Detection frame. dets: (N, >=5) [x1, y1, x2, y2, score, ...] in image pixels.
        Returns (M, 5) [x1, y1, x2, y2, track_id]."""
        d = np.asarray(dets, dtype=np.float64)[:, :5] if len(dets) else np.empty((0, 5))
        d = d.copy()  # OC-SORT rescales its input in place
        return self._remember(self.t.update(d, img_hw, img_hw))  # same sizes -> no rescaling

    def skip(self):
        """Skipped frame: Kalman-predicted boxes of the previously output tracks, (M, 5)."""
        return self._remember(self.t.skip())

    def advance(self, n, closed_form=True):
        """Multi-stream real-time mode: move the tracker over n dropped frames (never processed, no output).
        Same state change as n skipped frames, but last_boxes keeps the boxes of the last PROCESSED frame,
        so the NCC decision compares crops of that frame (the one kept as prev_gray) with the predictions.
        closed_form=True: one n-step Kalman prediction per track (Phase 4 O3) instead of n steps."""
        self.t.advance(n, closed_form)

    def peek_boxes(self):
        """For the skip decision, without changing tracker state:
        [(box output on the previous frame, Kalman prediction for the current frame)] per output track."""
        return [(self.last_boxes[t.id + 1], t.peek_predict())
                for t in self.t.output_tracks() if (t.id + 1) in self.last_boxes]
