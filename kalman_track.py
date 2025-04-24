from scipy.optimize import linear_sum_assignment
import scipy.linalg
import numpy as np
import os
import cv2
import time

np.random.seed(123)
# np.set_printoptions(suppress=True)


class KalmanFilter(object):
    """
    The 8-dimensional state space
    x, y, a, h, vx, vy, va, vh
    """
    count = 0
    def __init__(self, measurement):
        ndim, dt = 4, 1
        self.age = 0 # counter for no. of frame predicted
        self.hits = 0 # counter for no. of frame updated
        self.time_since_update = 0
        self.measurement_association = 0
        self.id = KalmanFilter.count
        KalmanFilter.count += 1

        self._motion_mat = np.eye(2*ndim, 2*ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim+i] = dt

        self._transform_mat = np.eye(ndim, 2*ndim)

        # Observation and Motion uncertainty are chosen relative to the
        # current state estimate. These weights control the amount of
        # uncertainty in the model.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

        # The mean vector(8 dimensional) and covariance matrix(8x8 dimensional)
        # of the new track. Unobserved velocities are initialized to 0.

        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)

        self.mean = np.r_[mean_pos, mean_vel]

        # print('Initial mean : {}'.format(self.mean))
        # print('*'*50)

        std = [self._std_weight_position * measurement[3],
               self._std_weight_position * measurement[3],
               1e-2,
               self._std_weight_position * measurement[3],
               self._std_weight_velocity * measurement[3],
               self._std_weight_velocity * measurement[3],
               1e-5,
               self._std_weight_velocity * measurement[3]]

        self.covariance = np.diag(np.square(std))

        # print('Initial Covariance : {}'.format(self.covariance))
        # print('*'*50)

    def predict(self):
        """
        Run the Kalman Filter Prediction step
        Return:
            The mean and covariance matrix of the predicted state.
            Unobserved velocities are initialized to 0 mean.
        """
        self.age += 1
        if self.time_since_update > 0:
            self.measurement_association = 0

        self.time_since_update += 1

        std_pos = [self._std_weight_position * self.mean[3],
                   self._std_weight_position * self.mean[3],
                   1e-2,
                   self._std_weight_position * self.mean[3]]

        std_vel = [self._std_weight_velocity * self.mean[3],
                   self._std_weight_velocity * self.mean[3],
                   1e-5,
                   self._std_weight_velocity * self.mean[3]]

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # print('Process Covariance(Error) : {}'.format(motion_cov))
        # print('*'*50)

        # x^(k+1) = F.x^k
        self.mean = np.dot(self._motion_mat, self.mean)
        # print('Predicted Mean : {}'.format(self.mean))
        # print('*'*50)
        # P^k = F.P^(k-1).Ft + Q
        self.covariance = np.linalg.multi_dot((self._motion_mat, self.covariance, self._motion_mat.T)) \
                            + motion_cov
        # print('Predicted Covariance : {}'.format(self.covariance))
        # print('*'*50)

        return np.dot(self._transform_mat, self.mean)

    def project(self):
        """
        Project state distribution to measurement state.
        Return: (ndarray, ndarray)
            The projected mean and covariance matrix of the given state estimate.
        """
        std = [self._std_weight_position * self.mean[3],
               self._std_weight_position * self.mean[3],
               1e-1,
               self._std_weight_position * self.mean[3]]

        measurement_cov = np.diag(np.square(std))

        # print('Measurement Error : {}'.format(measurement_cov))
        # print('*'*50)

        projected_mean = np.dot(self._transform_mat, self.mean)
        projected_cov = np.linalg.multi_dot((self._transform_mat, self.covariance, self._transform_mat.T)) \
                            + measurement_cov

        # print('Projected Mean : {}'.format(projected_mean))
        # print('*'*50)
        # print('Projected Covariance : {}'.format(projected_cov))
        # print('*'*50)

        return projected_mean, projected_cov

    def update(self, measurement):
        """
        Run KalmanFilter correction step.
        Returns:
            measurement-corrected state distribution.
        """
        self.time_since_update = 0
        self.hits += 1
        self.measurement_association += 1

        projected_mean, projected_cov = self.project()

        # (H.P.Ht + R).K' = P.Ht
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)

        # print('chol factor, lower : {}'.format(chol_factor))
        # print('*'*50)

        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower),
                                np.dot(self.covariance, self._transform_mat.T).T, check_finite=False).T
        # print('Kalman Gain : {}'.format(kalman_gain))
        # print('*'*50)

        pos_correction = measurement - projected_mean

        # print('Position Correction : {}'.format(pos_correction))
        # print('*'*50)

        self.mean = self.mean + np.dot(pos_correction, kalman_gain.T)

        # print('Updated mean : {}'.format(self.mean))
        # print('*'*50)

        self.covariance = self.covariance - np.linalg.multi_dot((
                    kalman_gain, projected_cov, kalman_gain.T))

        # print('Updated Covariance : {}'.format(self.covariance))
        # print('*'*50)

    def get_updated_state(self):
        return np.dot(self._transform_mat, self.mean)

def iou(d_bbox, t_bbox):
    # iou = np.sqrt((np.square(d_bbox[0] - t_bbox[0]) + \
    #         np.square(d_bbox[1] - t_bbox[1])))
    d_bbox[2] *= d_bbox[3]
    d_bbox[:2] -= d_bbox[2:4] / 2.
    d_bbox[2:4] += d_bbox[:2]

    t_bbox[2] *= t_bbox[3]
    t_bbox[:2] -= t_bbox[2:4] / 2.
    t_bbox[2:4] += t_bbox[:2]

    xx1 = np.maximum(d_bbox[0], t_bbox[0])
    yy1 = np.maximum(d_bbox[1], t_bbox[1])
    xx2 = np.minimum(d_bbox[2], t_bbox[2])
    yy2 = np.minimum(d_bbox[3], t_bbox[3])

    intersection_width = np.maximum(0., xx2 - xx1)
    intersection_height = np.maximum(0., yy2 - yy1)

    intersection_area = intersection_width * intersection_height

    union_area = ((d_bbox[2] - d_bbox[0]) * (d_bbox[3] - d_bbox[1])) + \
                 ((t_bbox[2] - t_bbox[0]) * (t_bbox[3] - t_bbox[1])) - \
                 intersection_area

    iou = intersection_area / union_area
    # print(iou)
    # print()

    return iou

def associate_detections_to_trackers(d_bbox, t_bbox, iou_threshold=0.2):

    if len(t_bbox) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(d_bbox)), np.empty((0, 5), dtype=int)

    iou_matrix = np.zeros((len(d_bbox), len(t_bbox)), dtype=np.float32)

    d_bbox_iou = d_bbox.copy()
    t_bbox_iou = t_bbox.copy()

    for d_idx, det in enumerate(d_bbox_iou):
        for t_idx, trk in enumerate(t_bbox_iou):
            iou_matrix[d_idx, t_idx] = iou(det.copy(), trk.copy())

    # print('IOU Matrix : {}'.format(iou_matrix))
    # print('*'*50)

    if iou_matrix.shape[0] == 1:

        if not np.any(iou_matrix > iou_threshold):
            # iou_matrix = np.delete(iou_matrix, 0, axis=0)
            matched_indices = np.empty(shape=(0, 2))

        else:
            argmax = np.argmax(iou_matrix)
            matched_indices = np.array([[0, argmax]])

    elif iou_matrix.shape[1] == 1:

        if not np.any(iou_matrix > iou_threshold):
            matched_indices = np.empty(shape=(0, 2))

        else:
            argmax = np.argmax(iou_matrix)
            matched_indices = np.array([[argmax, 0]])

    elif min(iou_matrix.shape) > 0:

        cost_matrix = -1 * iou_matrix # if you are using eucidean distance then dont't multiply with -1
        x, y = linear_sum_assignment(cost_matrix)
        # print('Applied Hungarian Algo.')
        # print('*'*50)
        matched_indices = np.array(list(zip(x, y)))
    else:
        matched_indices = np.empty(shape=(0, 2))

    # print('Matched Indices : {}'.format(matched_indices))
    # print('*'*50)

    unmatched_detections = []
    # print('d_bbox: {}'.format(d_bbox))
    for d_idx, det in enumerate(d_bbox):
        if (d_idx not in matched_indices[:, 0]):
            unmatched_detections.append(d_idx)

    unmatched_trackers = []
    # print('t_bbox: {}'.format(t_bbox))
    for t_idx, trk in enumerate(t_bbox):
        if (t_idx not in matched_indices[:, 1]):
            unmatched_trackers.append(t_idx)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matched_indices = np.concatenate(matches, axis=0)

    return matched_indices, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=30, min_ma=3):
        """
        Set key parameter for SORT
        """
        self.max_age = max_age # for how long we track the object when we stopped getting any measurement.
        self.min_ma = min_ma # new tracks are classified as tentative during first 3(min_ma) frame.
        #self.frame_count = 0 # frame count
        self.trackers = [] # trackers_list contains multiple tracked object in single frame.
        self.id_set = set()
        self.object_count = 0

    def update(self, detected_bbox=np.empty((0, 5))):
        """
        This method must be called once for each frame even with empty detections (use np.empty((0,5)) for frames without detections).

        Param:
            numpy array of detections in the format [[x1,y1,x2,y2,score],[x3,y3,x4,y4,score],...]
        Return:
            The similar array, where the last column is the object ID.
        NOTE:
            The number of objects returned may differ from the number of detections provided.
        """
        #self.frame_count += 1
        # if we are tracking, get predicted locations from existing trackers.
        # Initialize a tracked_bbox array, make prediction for each trackers and get the predicted position.
        tracked_bbox = np.zeros((len(self.trackers), 5))

        # print('Predicted box')
        # print()

        to_del = [] # contains invalid tracker index
        for tr_idx, each_tracker in enumerate(tracked_bbox):
            pos = self.trackers[tr_idx].predict()
            # print(pos)
            # check for invalid position
            if np.any(np.isnan(pos)):
                to_del.append(tr_idx)
                continue
            each_tracker[:] = [pos[0], pos[1], pos[2], pos[3], 0]

        for invalid_tr_idx in reversed(to_del):
            self.trackers.pop(invalid_tr_idx)
            np.delete(tracked_bbox, invalid_tr_idx, axis=0)

        d_bbox = detected_bbox.copy()
        t_bbox = tracked_bbox.copy()

        # print('Detected bbox: {}'.format(d_bbox))
        # print('*'*50)
        # print('Tracked bbox : {}'.format(t_bbox))
        # print('*'*50)

        matched, unmatched_dets, unmatched_tracks = associate_detections_to_trackers(d_bbox, t_bbox)

        # print('Matched : {}'.format(matched))
        # print('*'*50)
        # print('Unmatched_dets : {}'.format(unmatched_dets))
        # print('*'*50)
        # print('Unmatched_tracks : {}'.format(unmatched_tracks))
        # print('*'*50)

        # update matched trackers with detection bbox i.e, measurement
        for m in matched:
            self.trackers[m[1]].update(detected_bbox[m[0], :])

        # create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            new_tracker = KalmanFilter(detected_bbox[i, :])
            # print('Started Tracking...')
            self.trackers.append(new_tracker)

        # return only those tracker whose measurement association >= 3.
        # remove those tracker form tracker list whose time_since_update > max_age.
        ret = []
        i = len(self.trackers)
        #print(i)
        for each_tracker in reversed(self.trackers):

            d = each_tracker.get_updated_state()

            ret.append(np.concatenate((d, [each_tracker.id+1])).reshape(1, -1))

            if each_tracker.count not in self.id_set:
                if (each_tracker.measurement_association >= self.min_ma):
                    #ret.append(np.concatenate((d, [each_tracker.id+1])).reshape(1, -1))
                    self.id_set.add(each_tracker.count)

            if d[0] <= 100:
                i -= 1
                #self.id_set.remove(each_tracker.count)
                self.trackers.pop(i)
                # print('Deleted this track.')
                continue
            i -= 1
            # remove dead tracklet
            if (each_tracker.time_since_update > self.max_age):
                #self.id_set.remove(each_tracker.count)
                self.trackers.pop(i)

        if len(self.id_set):
            self.object_count = max(self.id_set)
            #print(self.id_set)

        if len(ret) > 0:
            return np.concatenate(ret), self.object_count

        return np.empty((0, 5)), self.object_count
