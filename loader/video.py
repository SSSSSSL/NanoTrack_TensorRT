import cv2

class Video:
    """ Video객체. 사실은 image의 연속입니다. """

    def __init__(self, path):
        self.path = path
        self.frames = []
        self.annotations = []

    def load_annotation(self, index):
        a = self.annotations[index]

        frame_num = a.frame_num
        bbox      = a.bbox

        return frame_num, cv2.imread(self.frames[frame_num]), bbox