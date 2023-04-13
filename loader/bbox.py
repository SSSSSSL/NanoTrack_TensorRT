
class BoundingBox:
    """ Bounding Box 클래스 """
    def __init__(self, x1, y1, x2, y2, frame_num):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.frame_num = frame_num

    def toXYWH(self):
        return [self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1]