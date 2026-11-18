"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import numpy as np
from .association import *
from .kalmanfilter import KalmanFilterNew


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age-dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h+1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0]+bbox1[2]) / 2.0, (bbox1[1]+bbox1[3])/2.0
    cx2, cy2 = (bbox2[0]+bbox2[2]) / 2.0, (bbox2[1]+bbox2[3])/2.0
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    # Pre-computed constant matrices shared across all trackers
    _F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [
                        0, 0, 0, 1, 0, 0, 0],  [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]], dtype=np.float64)
    _H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]], dtype=np.float64)

    def __init__(self, bbox, delta_t=3, orig=False):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        if not orig:
          self.kf = KalmanFilterNew(dim_x=7, dim_z=4)
        else:
          from filterpy.kalman import KalmanFilter
          self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = KalmanBoxTracker._F
        self.kf.H = KalmanBoxTracker._H

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t
        self.frames_since_obs = 0  # EMO: real frames (detect + skipped) since the last observation

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        if bbox is not None:
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age-dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)
            
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.frames_since_obs = 0  # EMO
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(bbox))
        else:
            self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        self.frames_since_obs += 1  # EMO
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    # ------------------------------------------------------------------ EMO additions
    def peek_predict(self):
        """EMO: box [x1,y1,x2,y2] the next predict() would give, without changing any state."""
        x = self.kf.x.copy()
        if (x[6] + x[2]) <= 0:
            x[6] *= 0.0
        return convert_x_to_bbox(np.dot(self.kf.F, x))[0]

    _multi_cache = {}  # EMO O3: n -> (F^n, sum_{j<n} F^j Q F^j.T), shared (all trackers use the same F and Q)

    def predict_skip_n(self, n, oru=True):
        """EMO O3: n consecutive predict_skip() calls in one closed-form step (frames dropped in real time).

        x_n = F^n x,  P_n = F^n P F^n.T + sum_{j<n} F^j Q F^j.T  - the same result as n sequential Kalman
        predictions up to float rounding. Falls back to the loop when predict()'s area clamp could fire
        within the n steps (shrinking box whose area would reach 0). With ORU active and the track observed,
        the first step stays sequential so that kf.update(None) freezes the state at the right frame.
        """
        if n <= 0:
            return
        x2, x6 = float(self.kf.x[2, 0]), float(self.kf.x[6, 0])
        if x2 + x6 <= 0 or (x6 < 0 and x2 + n * x6 <= 0):
            for _ in range(n):
                self.predict_skip(oru)
            return
        if oru and self.kf.observed:
            self.predict_skip(oru)  # freezes the KF at the first missing frame
            n -= 1
            if n == 0:
                return
        key = n
        if key not in KalmanBoxTracker._multi_cache:
            F, Q = self.kf.F, self.kf.Q
            Fn, Qs, Fj = np.eye(F.shape[0]), np.zeros_like(Q), np.eye(F.shape[0])
            for _ in range(n):
                Qs = Qs + Fj @ Q @ Fj.T
                Fj = F @ Fj
            Fn = Fj
            KalmanBoxTracker._multi_cache[key] = (Fn, Qs)
        Fn, Qs = KalmanBoxTracker._multi_cache[key]
        kf = self.kf
        kf.x = Fn @ kf.x
        kf.P = Fn @ kf.P @ Fn.T + Qs
        kf.x_prior, kf.P_prior = kf.x.copy(), kf.P.copy()
        kf.history_obs.extend([None] * n)
        if not kf.observed:
            # same bookkeeping as n x kf.update(None) on an already frozen (unobserved) filter
            kf.x_post, kf.P_post = kf.x.copy(), kf.P.copy()
            kf.z = np.array([[None] * kf.dim_z]).T
            kf.y = np.zeros((kf.dim_z, 1))
        # observed + fast path: predict_skip(oru=False) only appends None to history_obs
        self.age += n
        self.frames_since_obs += n
        self.history.append(convert_x_to_bbox(kf.x))

    def predict_skip(self, oru=True):
        """EMO: advance one frame on which detection was skipped.

        oru=False (Phase 4 fast path, O2): for a track that is currently observed, do not call
        kf.update(None) (which freezes the KF and makes the next detection replay the whole skipped run
        through ORU); only append None to the KF observation history so frame counting stays correct.
        A track matched again at the next detection frame then needs a single ordinary update.

        - Kalman state moves one step (same motion model as predict()).
        - kf.update(None) records "no observation this frame" in the KF history so that OC-SORT's
          ORU (freeze/unfreeze) later rebuilds the virtual trajectory with the true number of frames.
        - time_since_update / hit_streak are NOT touched: a skipped frame is not a missed detection,
          so the track keeps being output (fixes the V0 problem) and the detection-frame max_age clock
          is unchanged (V1). frames_since_obs counts real frames for the V2 clock.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        if oru or not self.kf.observed:
            self.kf.update(None)
        else:
            self.kf.history_obs.append(None)
        self.age += 1
        self.frames_since_obs += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {  "iou": iou_batch,
                "giou": giou_batch,
                "ciou": ciou_batch,
                "diou": diou_batch,
                "ct_dist": ct_dist}


class OCSort(object):
    def __init__(self, det_thresh, max_age=30, min_hits=3,
        iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2, use_byte=False, skip_mode="V1",
        skip_oru=True):
        """
        Sets key parameters for SORT
        skip_mode (EMO): how frames with skipped detection are handled, see skip().
        skip_oru (EMO): False = fast path of predict_skip (no ORU replay for tracks observed before the skip).
        """
        if skip_mode not in ("V0", "V1", "V2"):
            raise ValueError(f"skip_mode must be V0, V1 or V2, got {skip_mode}")
        self.skip_mode = skip_mode
        self.skip_oru = skip_oru
        self.last_output_ids = []  # EMO: tracker ids (0-based) output on the previous frame
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = use_byte
        KalmanBoxTracker.count = 0

    def update(self, output_results, img_info, img_size):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if output_results is None:
            return np.empty((0, 5))

        self.frame_count += 1
        # post_process detections
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        _zero_vel = np.array((0, 0))
        velocities = np.array(
            [trk.velocity if trk.velocity is not None else _zero_vel for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        """
            First round of association
        """
        matched, unmatched_dets, unmatched_trks = associate(
            dets, trks, self.iou_threshold, velocities, k_observations, self.inertia)
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        """
            Second round of associaton by OCR
        """
        # BYTE association
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = self.asso_func(dets_second, u_trks)          # iou between low score detections and unmatched tracks
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets_second[det_ind, :])
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.trackers[m].update(None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], delta_t=self.delta_t)
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                    this is optional to use the recent observation or the kalman filter prediction,
                    we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if self.skip_mode == "V2":
                # EMO V2: lost tracks age in real frames (skipped frames count toward max_age)
                dead = trk.time_since_update > 0 and trk.frames_since_obs > self.max_age
            else:
                # V0 / V1 (and no skipping): original OC-SORT rule
                dead = trk.time_since_update > self.max_age
            if dead:
                self.trackers.pop(i)
        self.last_output_ids = [int(r[0, 4]) - 1 for r in ret]  # EMO
        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

    # ------------------------------------------------------------------ EMO additions
    def advance(self, n, closed_form=True):
        """EMO: move every tracker over n frames that were dropped (real-time multi-stream). Same state as n
        skip() calls (V1/V2) but no outputs; closed_form=True uses predict_skip_n (O3), else the loop."""
        if n <= 0:
            return
        if self.skip_mode == "V0":
            for _ in range(n):
                self.update(np.empty((0, 5)), (1, 1), (1, 1))
            return
        self.frame_count += n
        for t in self.trackers:
            if closed_form:
                t.predict_skip_n(n, self.skip_oru)
            else:
                for _ in range(n):
                    t.predict_skip(self.skip_oru)

    def output_tracks(self):
        """EMO: trackers output on the previous frame (the set whose boxes EMO compares and propagates)."""
        ids = set(self.last_output_ids)
        return [t for t in self.trackers if t.id in ids]

    def skip(self):
        """EMO: process a frame on which detection is skipped.

        Returns (M, 5) [x1, y1, x2, y2, id] = Kalman-predicted boxes of the tracks output on the
        previous frame (EMO "use the predicted values as detections for frame f").

        V0 naive : the frame is an ordinary OC-SORT frame with no detections (update with an empty
                   array): time_since_update grows, hit_streak is reset at the next predict, the frame
                   counts toward max_age. Expected to suppress tracks after every skip.
        V1       : predict_skip(); max_age counts detection frames only (as FairMOT/EMO: frame_id is
                   not advanced on skipped frames).
        V2       : predict_skip(); max_age counts real frames for lost tracks (see update()).
        """
        prev = self.output_tracks()
        if self.skip_mode == "V0":
            self.update(np.empty((0, 5)), (1, 1), (1, 1))
            self.last_output_ids = [t.id for t in prev]  # keep propagating the same set while skipping
            alive = {t.id for t in self.trackers}
            boxes = [np.concatenate((t.get_state()[0], [t.id + 1])) for t in prev if t.id in alive]
        else:
            self.frame_count += 1
            boxes = [np.concatenate((t.predict_skip(self.skip_oru)[0], [t.id + 1])) for t in prev]
            # lost (not output) tracks also move one frame so their motion stays time-consistent
            for t in self.trackers:
                if t.id not in self.last_output_ids:
                    t.predict_skip(self.skip_oru)
        if boxes:
            return np.stack(boxes)
        return np.empty((0, 5))

    def update_public(self, dets, cates, scores):
        self.frame_count += 1

        det_scores = np.ones((dets.shape[0], 1))
        dets = np.concatenate((dets, det_scores), axis=1)

        remain_inds = scores > self.det_thresh
        
        cates = cates[remain_inds]
        dets = dets[remain_inds]

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            cat = self.trackers[t].cate
            trk[:] = [pos[0], pos[1], pos[2], pos[3], cat]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        _zero_vel = np.array((0, 0))
        velocities = np.array([trk.velocity if trk.velocity is not None else _zero_vel for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        matched, unmatched_dets, unmatched_trks = associate_kitti\
              (dets, trks, cates, self.iou_threshold, velocities, k_observations, self.inertia)
          
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
          
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            """
                The re-association stage by OCR.
                NOTE: at this stage, adding other strategy might be able to continue improve
                the performance, such as BYTE association by ByteTrack. 
            """
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_dets_c = left_dets.copy()
            left_trks_c = left_trks.copy()

            iou_left = self.asso_func(left_dets_c, left_trks_c)
            iou_left = np.array(iou_left)
            det_cates_left = cates[unmatched_dets]
            trk_cates_left = trks[unmatched_trks][:,4]
            cate_matrix = np.where(det_cates_left[:, np.newaxis] != trk_cates_left[np.newaxis, :], -1e6, 0.0)
            iou_left = iou_left + cate_matrix
            if iou_left.max() > self.iou_threshold - 0.1:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold - 0.1:
                          continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind) 
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            trk.cate = cates[i]
            self.trackers.append(trk)
        i = len(self.trackers)

        for trk in reversed(self.trackers):
            if trk.last_observation.sum() > 0:
                d = trk.last_observation[:4]
            else:
                d = trk.get_state()[0]
            if (trk.time_since_update < 1):
                if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
                    # id+1 as MOT benchmark requires positive
                    ret.append(np.concatenate((d, [trk.id+1], [trk.cate], [0])).reshape(1,-1)) 
                if trk.hit_streak == self.min_hits:
                    # Head Padding (HP): recover the lost steps during initializing the track
                    for prev_i in range(self.min_hits - 1):
                        prev_observation = trk.history_observations[-(prev_i+2)]
                        ret.append((np.concatenate((prev_observation[:4], [trk.id+1], [trk.cate], 
                            [-(prev_i+1)]))).reshape(1,-1))
            i -= 1 
            if (trk.time_since_update > self.max_age):
                  self.trackers.pop(i)
        
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0, 7))


