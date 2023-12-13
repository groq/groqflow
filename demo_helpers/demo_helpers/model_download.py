import os
import zipfile

from datasets.utils.file_utils import cached_path
from groqflow.common.build import DEFAULT_CACHE_DIR


YOLOV6N_MODEL = "yolov6n_model"
YOLOV6N_SOURCE = "yolov6n_source"


DATA_URLS = {
    YOLOV6N_MODEL: "https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6n.pt",
    YOLOV6N_SOURCE: "https://github.com/meituan/YOLOv6/archive/refs/tags/0.4.0.zip",
}


DST_PATHS = {
    YOLOV6N_MODEL: "pytorch_models/yolov6_nano/yolov6n.pt",
    YOLOV6N_SOURCE: "pytorch_models/yolov6_nano/YOLOv6",
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


def download_source(source):
    dst_path = os.path.join(DEFAULT_CACHE_DIR, DST_PATHS[source])
    if os.path.exists(dst_path):
        return dst_path

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    url = DATA_URLS[source]
    download_path = cached_path(url)
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        extracted_dir = os.path.dirname(dst_path)
        zip_ref.extractall(extracted_dir)
        os.rename(os.path.join(extracted_dir, zip_ref.infolist()[0].filename), dst_path)
    return dst_path
