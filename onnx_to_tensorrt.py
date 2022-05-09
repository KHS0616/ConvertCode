"""
Convert ONNX to TensorRT Code

Writer : KHS0616
Last Update : 2022-05-04
"""
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import glob, os
from PIL import Image

# TRT INT8 양자화 관련 변수
__all__ = [
    'PythonEntropyCalibrator',
    'ImageBatchStream'
]

def GiB(val):
    return val * 1 << 30

def set_onnx_option():
    option = {}
    option["model_path"] = "ECBNet.onnx"

    option["batch_size"] = 1
    option["channels"] = 3
    option["height"] = 1080
    option["width"] = 1920

    return option

def set_tensorrt_option():
    option = {}
    option["model_path"] = "output.engine"

    option["cali_path"] = "./DIV2K_test_HR"

    return option

def create_calibration_dataset(tensorrt_option):
    """ Calibration 데이터셋 생성 함수 """
    return glob.glob(os.path.join(tensorrt_option["cali_path"], '*.png'))


class PythonEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """ Int8 양자화를 위해 Calibrate를 진행하는 클래스 """
    def __init__(self, input_layers, stream, cache_file):
        super(PythonEntropyCalibrator, self).__init__()

        # Tensor RT에 지정될 Input 레이어 이름 설정
        self.input_layers = input_layers

        # Calib 이미지 배치 스트림 저장
        self.stream = stream

        # 데이터 GPU 할당
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)

        # 캐시파일 경로 저장
        self.cache_file = cache_file

        # 현재 스트림 리셋
        stream.reset()

    def get_batch_size(self):
        """ 배치사이즈 반환 메소드 """
        return self.stream.batch_size

    def get_batch(self, names):
        try:
            batch = self.stream.next_batch()
            if not batch.size:   
                return None

            cuda.memcpy_htod(self.d_input, batch)
            return [int(self.d_input)]

        except StopIteration:
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        # cache = ctypes.c_char_p(int(ptr))
        with open(self.cache_file, 'wb') as f:
            f.write(cache)

class ImageBatchStream():
    """ 양자화 과정에서 사용되는 이미지 배치 스트림 """
    def __init__(self, batch_size, calibration_files, WIDTH, HEIGHT, CHANNEL):
        # 배치사이즈 결정
        self.batch_size = batch_size
        self.max_batches = (len(calibration_files) // batch_size) + \
                        (1 if (len(calibration_files) % batch_size) \
                            else 0)

        # 파일 목록을 변수에 저장
        self.files = calibration_files

        # 변수 초기화
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.CHANNEL = CHANNEL
        self.calibration_data = np.zeros((batch_size, CHANNEL, HEIGHT, WIDTH), dtype=np.float32)

        # Calibration 배치 사이즈 설정
        self.batch = 0
     
    @staticmethod
    def read_image(path, WIDTH, HEIGHT, CHANNEL):
        """ 이미지를 읽고 전처리를 하는 메소드 """
        img = Image.open(path).convert('RGB').resize((WIDTH,HEIGHT), Image.BICUBIC)
        img = np.array(img, dtype=np.float32, order='C')
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img/255., axis=0)
        return img

    def reset(self):
        self.batch = 0
        
    def next_batch(self):
        if self.batch < self.max_batches:
            imgs = []
            # 배치사이즈에 따라 파일 목록 분류
            files_for_batch = self.files[self.batch_size * self.batch : self.batch_size * (self.batch + 1)]
                    
            for f in files_for_batch:
                print("[ImageBatchStream] Processing ", f)
                img = ImageBatchStream.read_image(f, self.WIDTH, self.HEIGHT, self.CHANNEL)
                imgs.append(img)
            for i in range(len(imgs)):
                self.calibration_data[i] = imgs[i]
            self.batch += 1

            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

def convert(onnx_info, tensorrt_info, int8_apply=False):
    TRT_LOGGER = trt.Logger()
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    # Calib 파일, 배치 스트림 함수, 양자화 함수 선언
    if int8_apply:
        calibration_files = create_calibration_dataset(tensorrt_info)
        batchstream = ImageBatchStream(1, calibration_files, onnx_info["width"], onnx_info["height"], onnx_info["channels"])
        calib = PythonEntropyCalibrator(["data"], batchstream, 'temp.cache')
    else:
        calib = False

    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    # 빌더, 네트워크, 파서를 통한 trt엔진 생성 과정
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network((EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        """ 빌드 생성, 네트워크 생성, onnx 파서 생성 """
        # ONNX 파일 읽기
        with open(onnx_info["model_path"], 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        print('Completed parsing of ONNX file')

        print('Network inputs:')

        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            print(tensor.name, trt.nptype(tensor.dtype), tensor.shape)

        config = builder.create_builder_config()
        config.max_workspace_size = GiB(2)  # 256MiB

        # 양자화 Calib 함수가 있으면 사용하도록 설정
        if calib:
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calib
        else:
            config.set_flag(trt.BuilderFlag.FP16)

        # # 동적일시 설정
        # profile = builder.create_optimization_profile()
        # profile.set_shape("input", (1, 3, 640, 640), (1, 3, 640, 640), (1, 3, 640, 640))
        # config.add_optimization_profile(profile)

        print('Building an engine from file {}; this may take a while...'.format(onnx_info["model_path"]))
        # trt엔진 빌드
        # engine = builder.build_engine(network, config)
        plan = builder.build_serialized_network(network, config)
        print("Completed creating Engine. Writing file to: {}".format(tensorrt_info["model_path"]))

        # 빌드된 trt 엔진을 저장
        with open(tensorrt_info["model_path"], "wb") as f:
            # f.write(engine.serialize())
            f.write(plan)

if __name__ == '__main__':
    onnx_option = set_onnx_option()
    tensorrt_option = set_tensorrt_option()

    convert(onnx_option, tensorrt_option, int8_apply=True)