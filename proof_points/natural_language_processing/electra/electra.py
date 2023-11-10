"""
The following example takes pre-trained ELECTRA small v2 from
huggingface models repository and executes against SST dataset on CPU
and GroqChip1 through GroqFlow.
"""
import os
import transformers
from groqflow import groqit
import torch
import numpy as np

from demo_helpers.compute_performance import compute_performance
from demo_helpers.args import parse_args


def evaluate_electra(rebuild_policy=None, should_execute=True):
    # set seed for consistency
    np.random.seed(1)
    torch.manual_seed(0)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # load pre-trained torch model
    pretrained_model_name = "howey/electra-base-sst2"

    tokenizer = transformers.ElectraTokenizerFast.from_pretrained(pretrained_model_name)
    pytorch_model = transformers.ElectraForSequenceClassification.from_pretrained(
        pretrained_model_name, torchscript=True
    )
    pytorch_model.eval()

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
    evaluate_electra(**parse_args())
