import os

from datasets.utils.file_utils import cached_path
from groqflow.common.build import DEFAULT_CACHE_DIR


YOLOV6N_ONNX = "yolov6n_onnx"


DATA_URLS = {
    YOLOV6N_ONNX: "https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6n.onnx",
}


DST_PATHS = {
    YOLOV6N_ONNX: "onnx_models/yolov6n.onnx",
}


def download_model(model):
    dst_path = os.path.join(DEFAULT_CACHE_DIR, DST_PATHS[model])
    if os.path.exists(dst_path):
        return dst_path

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    url = DATA_URLS[model]
    download_path = cached_path(url)
    os.symlink(download_path, dst_path)
    return dst_path
