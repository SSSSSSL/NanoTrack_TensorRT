import os
import glob

from loader.video import Video
from loader.bbox import BoundingBox

class VotLoader:
    """ VOT 데이터 로딩해주는 클래스\n
    {최상위 경로}/{ball,fish,hand...}/{*.jpg,groundtruth}의 형태로 배치되어있어야함
    """

    def __init__(self, path):
        """
        args:
            path(str): vot 데이터 최상위 경로
        """
        self.path = path
        self.videos = {}
        self.annotations = {}

        if not os.path.isdir(self.path):
            raise ValueError("디렉토리가 아닙니다~")

    
    def _get_subdir(self):
        """
        return:
            dirs(list): vot 데이터 최상위 경로 예하의 모든 디렉토리 반환
        """
        return [ dir for dir in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, dir)) ]


    def get_videos(self):
        """
        return:
            vot 경로에 존재하는 모든 video를 반환함
        """
        path = self.path
        dirs = self._get_subdir()

        for dir in dirs:
            video_path = glob.glob(os.path.join(path, dir, "*.jpg"))
            video = Video(video_path)
            frames = sorted(video_path)

            video.frames = frames

            gt_file = os.path.join(path, dir, 'groundtruth.txt')

            with open(gt_file, 'r') as f:
                for i, line in enumerate(f):
                    co_ords = line.strip().split(',')
                    co_ords = [(float(co_ord)) for co_ord in co_ords]
                    ax, ay, bx, by, cx, cy, dx, dy = co_ords
                    x1 = min(ax, min(bx, min(cx, dx))) - 1
                    y1 = min(ay, min(by, min(cy, dy))) - 1
                    x2 = max(ax, max(bx, max(cx, dx))) - 1
                    y2 = max(ay, max(by, max(cy, dy))) - 1
                    bbox = BoundingBox(x1, y1, x2, y2, i)
                    video.annotations.append(bbox)
            
            self.videos[dir] = [video.frames, video.annotations]

        return self.videos



