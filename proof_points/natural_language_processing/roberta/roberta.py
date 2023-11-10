"""
The following example takes a pre-trained RoBERTa model and executes
against CoNNL 2003 dataset on CPU and GroqChip™ processor by using
the GroqFlow toolchain. The model and data set can be downloaded
here: https://huggingface.co/dominiqueblok/roberta-base-finetuned-ner
"""

import os

import torch

from demo_helpers.compute_performance import compute_performance
from demo_helpers.args import parse_args
from groqflow import groqit
from transformers import RobertaForTokenClassification, RobertaTokenizerFast


def evaluate_roberta(rebuild_policy=None, should_execute=None):
    # set seed for consistency
    torch.manual_seed(0)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # load pre-trained torch model
    model_path = "dominiqueblok/roberta-base-finetuned-ner"
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    torch_model = RobertaForTokenClassification.from_pretrained(
        model_path, torchscript=True
    )

    # dummy inputs to generate the groq model
    batch_size, max_seq_length = 1, 128
    dummy_inputs = {
        "input_ids": torch.ones((batch_size, max_seq_length), dtype=torch.long),
        "attention_mask": torch.ones((batch_size, max_seq_length), dtype=torch.float),
    }

    # generate groq model
    build_name = "roberta"
    groq_model = groqit(
        torch_model,
        dummy_inputs,
        compiler_flags=["--large-program"],
        rebuild=rebuild_policy,
        build_name=build_name,
    )

    # compute performance on CPU and GroqChip
    if should_execute:
        compute_performance(
            groq_model,
            torch_model,
            dataset="conll2003",
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            task="ner",
        )

    print(f"Proof point {__file__} finished!")


if __name__ == "__main__":
    evaluate_roberta(**parse_args())
