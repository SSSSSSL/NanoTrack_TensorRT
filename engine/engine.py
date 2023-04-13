import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # 없으면, "invalid device context - no currently active context?" 에러 뜸

from utils.timer import timer

TRT_LOGGER = trt.Logger()

class TRTEngine(object):
    """ TensorRT 엔진 랩퍼 클래스\n
    마치 torch의 nn.Module처럼 callable한 클래스\n
    단, forward는 지원하지만 backward는 안됨\n
    그리고 general하게 만들어두진 않았음!"""
    
    def __init__(self, engine_file_path=""):
        """ TRT Engine을 가져와서, 직렬화하여 로딩 """

        if (os.path.exists(engine_file_path)):
            print("Loading engine from path {}".format(engine_file_path))
            with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
                self.context = self.engine.create_execution_context()
                self._allocate() # 데이터 할당
        else:
            raise FileNotFoundError('Engine file {} is not found'.format(engine_file_path))


    def __call__(self, data):
        """ 인스턴스를 callable하게 만드는 함수\n
        결과가 항상 list에 담긴다는 것에 유의할 것 """

        # head에서 처리할 때, input이 2개
        if isinstance(data, list):
            if len(data) != 2:
                raise ValueError("head의 입력은 2개입니다")
            
            # data[0] -> tensor(compatible with tensorrt)
            # data[1] -> array of image

            # data[0] (zf) -> 1 * 48 * 8  * 8  =  3,072
            # data[1] (xf) -> 1 * 48 * 16 * 16 = 12,288

            # output[0]    -> 1 * 2 * 16 * 16  = 512
            # output[1]    -> 1 * 4 * 16 * 16  = 1024

            # 처리할 데이터를 입력 host에 저장
            self.inputs[0].host = data[0]
            self.inputs[1].host = data[1]

            # input 데이터를 GPU로 보낸다
            [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]

            # 추론을 한다
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

            # 처리된 데이터를 host로 가져온다
            [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]

            # 동기화한다
            self.stream.synchronize()

            # 결과를 반환한다
            return [out.host for out in self.outputs]
        else:
            # templar image일 때 
            # data[0] (z) ->   1 * 3 * 127 * 127 = 48,387

            # exampler image일 때
            # data[0] (x) ->   1 * 3 * 255 * 255 = 195,075
            
            # 처리할 데이터를 입력 host에 저장
            self.inputs[0].host = data

            # input 데이터를 GPU로 보낸다
            [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]

            # 추론을 한다
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

            # 처리된 데이터를 host로 가져온다
            [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]

            # 동기화한다
            self.stream.synchronize()

            # 결과를 반환한다
            return [out.host for out in self.outputs]
        

    def _allocate(self):
        """ TRT Engine에 필요한 데이터를 할당\n
        데이터를 cpu, gpu에 각각 올리기 위한 함수 """
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            self.bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem, dtype))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem, dtype))


    def get_input_dtype(self):
        """ input에 대한 dtype을 반환하는 함수\n
        args:
            void
        return:
            dtypes(list)
        """
        return [ inp.dtype for inp in self.inputs ]


class HostDeviceMem(object):
    """ 데이터 3개(host mem, device mem, dtype)을 담기위한 보조 클래스 """
    def __init__(self, host_mem, device_mem, dtype):
        self.host = host_mem
        self.device = device_mem
        self.dtype = dtype

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device) + "\nDtype:\n" + str(self.dtype)

    def __repr__(self):
        return self.__str__()