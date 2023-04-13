# 일단 구조부터 생각하자
# 기존의 구조를 그대로 가져가고 싶음. 단, trt engine 따로, image 처리 등을 위한 객체 따로...
# trt를 실행시키는 방법부터 고민해보자.

import cv2

from tracker.tracker import Tracker
from loader.vot import VotLoader

def get_frames(video_name, resolution="FHD"):
    preset = {
        "UHD": (3840, 2160),
        "QHD": (2560, 1440),
        "FHD": (1920, 1080),
        "HD":  (1280, 720),
        "VGA": (640, 480)
    }

    if isinstance(video_name, int):
        cap = cv2.VideoCapture(video_name)
    elif video_name == 'webcam':
        cap = cv2.VideoCapture(0)

    width, height = preset[resolution]

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # warm up
    for i in range(5):
        cap.read()
    
    while True:
        ret, frame = cap.read()
        timer = cv2.getTickCount()
        if ret:
            yield frame, timer
        else:
            break


def trackLive(tracker):
    video_name = 0
    first = True

    for frame, timer in get_frames(video_name):
        # init
        if first: 
            try:
                init_bbox = cv2.selectROI('Select ROI', frame, False, False)
                cv2.destroyAllWindows()
            except:
                exit()
            
            tracker.init(frame, init_bbox)
            first = False
        # track
        else:
            outputs = tracker.track(frame)

            bbox = list(map(int, outputs['bbox']))
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                                 (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                 (0, 255, 0), 3)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        # show
        cv2.imshow('Tracking', frame)
        cv2.waitKey(1)


def trackVideo(tracker, video_path, save_video=False):
    cap = cv2.VideoCapture(video_path)
    first = True

    if save_video:
        print('Save Video Path : ', './out.mp4')
        outVideo = cv2.VideoWriter('./out.mp4',
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   int(cap.get(cv2.CAP_PROP_FPS)),
                                   (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timer = cv2.getTickCount()

        if first:
            try:
                init_bbox = cv2.selectROI('Select ROI', frame, False, False)
                cv2.destroyAllWindows()
            except:
                exit()
            
            tracker.init(frame, init_bbox)
            first = False
        else:
            outputs = tracker.track(frame)

            bbox = list(map(int, outputs['bbox']))
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                                 (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                 (0, 255, 0), 3)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        # show
        cv2.imshow('Tracking', frame)
        cv2.waitKey(1)

        # save
        if save_video:
            outVideo.write(frame)

    if save_video:
        outVideo.release()


def benchmarkVOT(tracker, path):

    vot = VotLoader(path)
    videos = vot.get_videos()
    keys = list(videos.keys())

    for i in range(0, len(videos)):
        video_frames = videos[keys[i]][0]
        annot_frames = videos[keys[i]][1]

        num_frames = min(len(video_frames), len(annot_frames))

        frames_0 = video_frames[0]
        bbox_0   = annot_frames[0]
        img = cv2.imread(frames_0)

        # init
        # convert x1, y1, x2, y2 -> x, y, w, h
        tracker.init(img, [bbox_0.x1, bbox_0.y1, bbox_0.x2 - bbox_0.x1, bbox_0.y2 - bbox_0.y1])

        for j in range(1, num_frames):
            frame = video_frames[j]
            bbox  = annot_frames[j]

            img = cv2.imread(frame)
            imgCopy = img.copy()
            timer = cv2.getTickCount()

            # draw ground truth
            cv2.rectangle(imgCopy, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), (255, 255, 255), 2)
            
            # track
            outputs = tracker.track(img)

            pred_bbox = list(map(int, outputs['bbox']))

            # draw predicted
            cv2.rectangle(imgCopy, (pred_bbox[0], pred_bbox[1]),
                                   (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]),
                                   (255, 0, 0), 2)

            # calculate fps
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            cv2.putText(imgCopy, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

            # show
            cv2.imshow('VOT Benchmark', imgCopy)
            cv2.waitKey(30)


def main():
    """ TensorRT 엔진을 로딩하고, 추론하는 메인 함수 """
    back_exam_engine_path = "engine/nanotrack_backbone_exam.engine"
    back_temp_engine_path = "engine/nanotrack_backbone_temp.engine"
    head_engine_path = "engine/nanotrack_head.engine"

    tracker = Tracker(back_exam_engine_path, back_temp_engine_path, head_engine_path)

    # webcam 트래킹 시작
    # trackLive(tracker)


    # video 트래킹 시작
    video_path = "/home/rgblab/crop2.mp4"
    trackVideo(tracker, video_path, save_video=False)


    # benchmark 시작
    # vot_path = "/home/rgblab/tracking/dataset/VOT"
    # benchmarkVOT(tracker, vot_path)


if __name__ == '__main__':
    main()
