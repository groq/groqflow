"""
The following example takes pre-trained Bert from Hugging Face
(https://huggingface.co/howey/bert-base-uncased-sst2) and
executes against SST dataset (https://paperswithcode.com/dataset/sst)
on CPU and GroqCard™ processor using the GroqFlow toolchain.
"""
import os
import numpy as np
import torch
import transformers
from groqflow import groqit

from demo_helpers.compute_performance import compute_performance
from demo_helpers.args import parse_args


def get_model():
    """PyTorch Model setup."""
    pretrained_model_name = "howey/bert-base-uncased-sst2"

    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name)
    pytorch_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name, torchscript=True
    )

    return pytorch_model.eval(), tokenizer


def evaluate_bert(rebuild_policy=None, should_execute=True):
    # set seed for consistency
    np.random.seed(1)
    torch.manual_seed(0)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # load pre-trained torch model
    pytorch_model, tokenizer = get_model()

    # dummy inputs to generate the groq model
    batch_size = 1
    max_seq_length = 128
    dummy_inputs = {
        "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.bool),
    }

    # generate groq model
    groq_model = groqit(pytorch_model, dummy_inputs, rebuild=rebuild_policy)

    # compute performance on CPU and GroqChip
    if should_execute:
        compute_performance(
            groq_model,
            pytorch_model,
            dataset="sst",
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            task="classification",
        )

    print(f"Proof point {__file__} finished!")


if __name__ == "__main__":
    evaluate_bert(**parse_args())
