"""
The following example downloads a pre-trained DeiT tiny from Hugging
Face (https://huggingface.co/facebook/deit-tiny-patch16-224) and
executes against Imagenette, the 10-class, sampled ImageNet
dataset (https://github.com/fastai/imagenette) on CPU and
GroqChip™ processor by using the GroqFlow toolchain.
"""
import torch
from transformers import ViTForImageClassification

from groqflow import groqit
from demo_helpers.compute_performance import compute_performance
from demo_helpers.args import parse_args


def evaluate_deit_tiny(rebuild_policy=None, should_execute=True):
    # load torch model
    model = ViTForImageClassification.from_pretrained("facebook/deit-tiny-patch16-224")
    model.eval()

    # create dummy inputs to prime groq model
    dummy_inputs = {"pixel_values": torch.ones([1, 3, 224, 224])}

    # generate groq model
    groq_model = groqit(model, dummy_inputs, rebuild=rebuild_policy)

    # compute performance on CPU and GroqChip
    if should_execute:
        return compute_performance(
            groq_model,
            model,
            dataset="sampled_imagenet",
            task="classification",
        )


if __name__ == "__main__":
    evaluate_deit_tiny(**parse_args())
