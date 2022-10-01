import os
import numpy as np
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch
from demo_helpers.compute_performance import compute_performance
from demo_helpers.args import parse_args

from groqflow import groqit


def evaluate_distilbert(rebuild_policy=None, should_execute=True):
    # set seed for consistency
    np.random.seed(1)
    torch.manual_seed(0)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # load pre-trained torch model
    pretrained_model = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    pytorch_model = DistilBertForSequenceClassification.from_pretrained(
        pretrained_model
    )

    # dummy inputs to generate the groq model
    batch_size = 1
    max_seq_length = 128

    dummy_inputs = {
        "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.bool),
    }

    # generate groq model
    build_name = "distilbert"
    groq_model = groqit(
        pytorch_model, dummy_inputs, rebuild=rebuild_policy, build_name=build_name
    )

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


if __name__ == "__main__":
    evaluate_distilbert(**parse_args())
