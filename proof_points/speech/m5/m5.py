"""
The following example takes a pre-trained M5 model and executes against
SpeechCommands dataset on CPU and GroqChip™ processor using Groqflow.
"""

import torch

from demo_helpers.compute_performance import compute_performance
from demo_helpers.models import load_pretrained
from demo_helpers.args import parse_args
from groqflow import groqit


def evaluate_m5(rebuild_policy=None, should_execute=True):
    # set seed for consistency
    torch.manual_seed(0)

    # load pre-trained torch model
    torch_model = load_pretrained("m5")
    torch_model.eval()

    # dummy inputs to generate groq model
    dummy_input = torch.randn([1, 1, 16000])

    # generate groq_model
    build_name = "m5"
    groq_model = groqit(
        torch_model, {"x": dummy_input}, rebuild=rebuild_policy, build_name=build_name
    )

    # compute performance on CPU, GroqChip
    if should_execute:
        compute_performance(
            groq_model,
            torch_model,
            dataset="speechcommands",
            task="keyword_spotting",
        )


if __name__ == "__main__":
    evaluate_m5(**parse_args())
