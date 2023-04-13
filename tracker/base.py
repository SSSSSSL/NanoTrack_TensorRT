import numpy as np
import cv2

# from utils.timer import timer
# from utils.timer import StopWatch

class BaseTracker(object):
    """ Tracker에 대한 Interface 클래스 """

    def init(self, img, bbox):
        """
        img(np.ndarray): BGR image
        bbox: Bounding Box => [x, y, w, h]
        """
        raise NotImplementedError

    def track(self, img):
        """
        img(np.ndarray): BGR image
        bbox: Bounding Box => [x, y, w, h]
        """
        raise NotImplementedError

class SiameseTracker(BaseTracker):
    # BottleNeck Function... Time Consumer...
    # 시간 오래 잡아먹어요...
    # @timer
    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        im: BGR image
        pos: center position
        model_sz: exemplar size
        original_sz: original size
        avg_chans: channel average

        return: np.ndarray, not tensor
        """
        if isinstance(pos, float):
            pos = [pos, pos]

        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad
        
        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            ## 시간 많이 잡아먹어요~~ 
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad: 
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad: 
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                             int(context_xmin):int(context_xmax + 1), :]
            ## 시간 많이 잡아먹어요~~ 
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                          int(context_xmin):int(context_xmax + 1), :]
        
        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        # im_patch = torch.from_numpy(im_patch)
        # if cfg.CUDA:
            # im_patch = im_patch.cuda()
        return im_patch