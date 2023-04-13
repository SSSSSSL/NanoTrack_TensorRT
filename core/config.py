import numpy as np

class Config:
    """ Config 정보 모아둔 PO클래스 """
    TRACK_CONTEXT_AMOUNT     = 0.5
    TRACK_EXEMPLAR_SIZE      = 127
    TRACK_INSTANCE_SIZE      = 255
    TRACK_OUTPUT_SIZE        = 16

    TRACK_PENALTY_K          = 0.16
    TRACK_WINDOW_INFLUENCE   = 0.46
    TRACK_LR                 = 0.34

    POINT_STRIDE             = 8

config = Config