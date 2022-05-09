"""
Convert ONNX to TensorFlow Code

Writer : KHS0616
Last Update : 2022-05-09
"""

import tensorflow as tf

cali_path = "./DIV2K_test_HR"

def set_option():
    option = {}

    option["tensorflow_path"] = ""
    option["tensorflowlite_path"] = ""

    return option

def makeCalibrationDataset():
    """ Calibration 데이터 셋 생성 메소드 """
    cali_datasets = []
    for v in os.listdir(cali_path):
        img = cv2.imread(os.path.join(cali_path, v)).astype(np.float32)
        img = cv2.resize(img, (1280,720), cv2.INTER_CUBIC)
        img = img.transpose(2, 0, 1) / 255.
        img = np.expand_dims(img[0], 0)
        # img = tf.io.read_file(os.path.join(self.opt["tensorlite"]["cali_path"], v))
        # img = tf.image.decode_jpeg(img)
        # img = tf.image.random_crop(img, [128, 128, 3], seed=None, name=None)
        # img = tf.cast(img, tf.float32) / 255.0
        cali_datasets.append(img)
        break
    return cali_datasets

def representative_data_gen():
    """ TensorLite 버전 Calibration 메소드 """ 
    cali_datasets = makeCalibrationDataset()
    for input_value in tf.data.Dataset.from_tensor_slices(cali_datasets).batch(1).take(100):
        # Model has only one input so each data point has one element.
        yield [input_value]

def maketflite(opt, int8_apply=False):
    """ Tensorflow PB 파일을 TFLite 파일로 변환하는 함수 """
    pb_path = opt["tensorflow_path"]
    converter = tf.lite.TFLiteConverter.from_saved_model(pb_path)

    # 에러 방지
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

    if int8_apply:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # converter.target_spec.supported_types = [tf.float16]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    tf_lite_model = converter.convert()
    open(opt["tensorflowlite_path"], "wb").write(tf_lite_model)

if __name__ == '__main__':
    option = set_option()
    convert(option, int8_apply=True)