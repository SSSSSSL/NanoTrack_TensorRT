import numpy as np
import torch

class BackBoneProcessor(object):
    """ Tracker, BackBone 네트워크의 input과 output에 대해서 전후처리 해주는 클래스\n
    입출력에 맞는 형변환 및 데이터 변환을 수행함 """

    def __init__(self, input_dtype):
        if isinstance(input_dtype, list):
            self.in_dtype = input_dtype[0]
        else:
            self.in_dtype = input_dtype

        
    def pre(self, input):
        # 현재는 일단 nd array를 1d array로 변경만 함
        return np.ravel(input)
        # return np.array(input, self.in_dtype)

    def post(self, output):
        # 항상 list로 반환받기에, (engine에서 항상 list로 반환))
        if len(output) != 1:
            raise ValueError("engine의 타입은 list지만, 갯수는 1개여야함!")

        return output[0]


class HeadProcessor(object):
    """ Tracker, Head 네트워크의 input과 output에 대해서 전후처리 해주는 클래스\n
    입출력에 맞는 형변환 및 데이터 변환을 수행함 """

    def __init__(self, input_dtype):
        if isinstance(input_dtype, list):
            self.in_dtype = input_dtype[0]
        else:
            self.in_dtype = input_dtype

    def pre(self, input1, input2):
        return [input1, input2]

    def post(self, output1, output2):
        # 일단 torch.tensor로 변경
        # output1(zf) -> 1 * 2 * 16 * 16 -> cls
        # output2(xf) -> 1 * 4 * 16 * 16 -> loc
        output1 = torch.from_numpy(output1)
        output2 = torch.from_numpy(output2)

        output1 = torch.reshape(output1, [1,2,16,16])
        output2 = torch.reshape(output2, [1,4,16,16])

        return output1, output2