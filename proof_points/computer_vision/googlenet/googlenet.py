"""
The following example takes a pre-trained GoogLeNet model
(https://pytorch.org/hub/pytorch_vision_googlenet/) and
executes against Imagenette, the 10-class, sampled ImageNet
dataset (https://github.com/fastai/imagenette) on CPU and
GroqChip™ processor by using the GroqFlow toolchain.
"""
import torch

from demo_helpers.compute_performance import compute_performance
from demo_helpers.args import parse_args
from groqflow import groqit


def evaluate_googlenet(rebuild_policy=None, should_execute=None):
    # set seed for consistency
    torch.manual_seed(0)

    # load torch model
    torch_model = torch.hub.load(
        "pytorch/vision:v0.10.0", "googlenet", weights="GoogLeNet_Weights.DEFAULT"
    )
    torch_model.eval()  # disable training specific layers

    # create dummy input to prime groq model
    dummy_inputs = torch.randn((1, 3, 224, 224), dtype=torch.float32)

    # generate groq model
    build_name = "googlenet"
    groq_model = groqit(
        torch_model,
        {"x": dummy_inputs},
        rebuild=rebuild_policy,
        build_name=build_name,
    )

    # compute performance on CPU and GroqChip
    if should_execute:
        compute_performance(
            groq_model, torch_model, "sampled_imagenet", task="classification"
        )

    print(f"Proof point {__file__} finished!")


if __name__ == "__main__":
    evaluate_googlenet(**parse_args())
