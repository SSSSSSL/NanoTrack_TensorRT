import tensorrt as trt
import numpy as np

from core.config import config

from tracker.model import Model
from tracker.base import SiameseTracker

from utils.bbox import corner2center

#  Tracker에서 추론을 제외한 모든 tracking 로직이 구현됨

class Tracker(SiameseTracker):
    """ TensorRT 엔진을 활용해, Tracking을 하는 객체 """

    def __init__(self, back_exam_engine_path="", temp_exam_engine_path="", head_engine_path=""):
        self.score_size = config.TRACK_OUTPUT_SIZE

        hanning = np.hanning(self.score_size)
        window  = np.outer(hanning, hanning)
        self.cls_out_channels = 2
        self.window = window.flatten()

        self.points = self.generate_points(config.POINT_STRIDE, self.score_size)
        self.model = Model(back_exam_engine_path, temp_exam_engine_path, head_engine_path) # TRT engine 모델 생성

    def generate_points(self, stride, size):
        ori = - (size // 2) * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()

        return points

    def _convert_bbox(self, delta, point):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.detach().cpu().numpy()

        delta[0, :] = point[:, 0] - delta[0, :] #x1
        delta[1, :] = point[:, 1] - delta[1, :] #y1
        delta[2, :] = point[:, 0] + delta[2, :] #x2
        delta[3, :] = point[:, 1] + delta[3, :] #y2
        delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)
        return delta

    def _convert_score(self, score):
        if self.cls_out_channels == 1:
            score = score.permute(1, 2, 3, 0).contiguous().view(-1)
            score = score.sigmoid().detach().cpu().numpy()
        else:
            score = score.permute(1, 2, 3, 0).contiguous().view(self.cls_out_channels, -1).permute(1, 0)
            score = score.softmax(1).detach()[:, 1].cpu().numpy()
        return score        

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: Bounding Box => [x, y, w, h]
        return:
            void
        """
        # center_pos = [center x of bbox, center y of bbox]
        self.center_pos = np.array([bbox[0] + (bbox[2]-1) / 2,
                                    bbox[1] + (bbox[3]-1) / 2])

        # size = [w, h]
        self.size = np.array([bbox[2], bbox[3]])

        # z crop size 계산
        w_z = self.size[0] + config.TRACK_CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + config.TRACK_CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # img channel average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get z crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    config.TRACK_EXEMPLAR_SIZE,
                                    s_z, self.channel_average)

        # forward only backbone
        self.model.template(z_crop)

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list): [x, y, width, height]
        """
        w_z = self.size[0] + config.TRACK_CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + config.TRACK_CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        scale_z = config.TRACK_EXEMPLAR_SIZE / s_z
        s_x = s_z * (config.TRACK_INSTANCE_SIZE / config.TRACK_EXEMPLAR_SIZE)

        # get x crop 
        x_crop = self.get_subwindow(img, self.center_pos,
                                    config.TRACK_INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        # forward all (backbone, head)
        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        bbox  = self._convert_bbox(outputs['loc'], self.points)


        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(bbox[2, :], bbox[3, :]) /
                    (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (bbox[2, :] / bbox[3, :]))

        penalty = np.exp(-(r_c * s_c - 1) * config.TRACK_PENALTY_K)

        # score
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - config.TRACK_WINDOW_INFLUENCE) + \
            self.window * config.TRACK_WINDOW_INFLUENCE

        best_idx = np.argmax(pscore)

        bbox = bbox[:, best_idx] / scale_z

        lr = penalty[best_idx] * score[best_idx] * config.TRACK_LR
        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width  = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        # update state
        self.center_pos = np.array([cx, cy])
        self.size       = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        best_score = score[best_idx]

        return {
                'bbox': bbox,
                'best_score': best_score
               }