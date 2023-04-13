from engine.engine import TRTEngine

from core.process import BackBoneProcessor
from core.process import HeadProcessor

# base.py에서 구현한 Tracker 모델을 구체화 함
# base.py의 BaseTracker와 SiameseTracker는 정확히는 트래킹을 위한 헬퍼 함수를 가지고 있는 상태임.
# 실제 추론은 Engine에서 이뤄지며 Model은 Engine과 Tracker를 묶어주는 역할

class Model(object):
    """ TensorRT Engine을 통해 데이터를 추론하는 객체 """

    def __init__(self, exam_back_engine_path="", temp_back_engine_path="", head_engine_path=""):
        self.back_exam_engine = TRTEngine(exam_back_engine_path)
        self.back_temp_engine = TRTEngine(temp_back_engine_path)
        self.head_engine      = TRTEngine(head_engine_path)

        # 입력과 출력에 대한 차원 변환을 수행해주는 클래스
        self.back_exam_processor = BackBoneProcessor(self.back_exam_engine.get_input_dtype())
        self.back_temp_processor = BackBoneProcessor(self.back_temp_engine.get_input_dtype())
        self.head_processor      = HeadProcessor(self.head_engine.get_input_dtype())
        

    def template(self, z):
        """
        args:
            z(ndarray): BGR image
        return:
            void
        """
        zf = self.back_temp_engine(self.back_temp_processor.pre(z))
        self.zf = self.back_temp_processor.post(zf)


    def track(self, x):
        """
        args:
            x(ndarray): BGR image
        return:
            {'cls': cls, 'loc': loc}
        """
        # x에 대해서 exam백본망 들어가기 전, 전처리
        xf = self.back_exam_engine(self.back_exam_processor.pre(x))
        
        # xf에 대해서 exam백본망 후처리 후,
        # head망 전처리를 함
        cls, loc = self.head_engine(self.head_processor.pre(self.zf, self.back_exam_processor.post(xf)))

        # head망 후처리 진행
        cls, loc = self.head_processor.post(cls, loc)

        return {
                'cls': cls,
                'loc': loc,
               }
