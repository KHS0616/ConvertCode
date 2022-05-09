"""
Convert ONNX to TensorFlow Code

Writer : KHS0616
Last Update : 2022-05-09
"""
import os
import onnx
from onnx_tf.backend import prepare

def set_option():
    option = {}
    option["onnx_path"] = ""
    option["tf_path"] = ""

    return option

def convert(opt):
    # ONNX 모델을 불러오기
    onnx_model = onnx.load(opt["onnx_path"])
    tf_rep = prepare(onnx_model)

    # pb파일 경로 설정
    pb_path = opt["tf_path"]

    # pb 파일로 Converting
    tf_rep.export_graph(pb_path)

if __name__ == '__main__':
    option = set_option()
    convert(option)