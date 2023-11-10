"""
The following example takes pre-trained YOLOv6 model
(https://github.com/meituan/YOLOv6) and executes against
the COCO dataset (https://cocodataset.org/) on CPU and
GroqChip™ processor using the GroqFlow toolchain.
"""

from groqflow import groqit
from demo_helpers.args import parse_args
from demo_helpers.compute_performance import compute_performance
from demo_helpers.model_download import YOLOV6N_ONNX, download_model
from demo_helpers.misc import check_deps

import torch


def get_onnx_model():
    return download_model(YOLOV6N_ONNX)


def evaluate_yolov6n(rebuild_policy=None, should_execute=True):
    check_deps(__file__)
    pytorch_model = get_onnx_model()
    dummy_inputs = {"image_arrays": torch.ones([1, 3, 640, 640])}

    # Get Groq Model using groqit
    groq_model = groqit(
        pytorch_model,
        dummy_inputs,
        rebuild=rebuild_policy,
    )
    if should_execute:
        compute_performance(groq_model, pytorch_model, "coco", task="coco_map")

    print(f"Proof point {__file__} finished!")


if __name__ == "__main__":
    evaluate_yolov6n(**parse_args())
