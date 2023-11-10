"""
The following example takes pre-trained ResNet-50 from torchvision and executes against
Imagenet1k on CPU and GroqChip1 through GroqFlow.
"""

from groqflow import groqit
from demo_helpers.args import parse_args
from demo_helpers.compute_performance import compute_performance

import torch


def get_model():
    """PyTorch Model setup."""
    pytorch_model = torch.hub.load(
        "pytorch/vision:v0.10.0", "resnet50", weights="ResNet50_Weights.IMAGENET1K_V1"
    )
    return pytorch_model.eval()


def evaluate_resnet50(rebuild_policy=None, should_execute=True):
    pytorch_model = get_model()
    dummy_inputs = {"x": torch.ones([1, 3, 224, 224])}

    # Get Groq Model using groqit
    groq_model = groqit(pytorch_model, dummy_inputs, rebuild=rebuild_policy)

    # Execute PyTorch model on CPU, Groq Model and print accuracy
    if should_execute:
        compute_performance(
            groq_model, pytorch_model, "sampled_imagenet", task="classification"
        )

    print(f"Proof point {__file__} finished!")


if __name__ == "__main__":
    evaluate_resnet50(**parse_args())
