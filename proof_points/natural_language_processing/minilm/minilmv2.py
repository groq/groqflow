"""
The following example takes pre-trained MiniLM v2 from
huggingface models repository and executes against STS benchmark dataset
on CPU and GroqChip1 through GroqFlow.
"""
import os
from transformers import AutoTokenizer, AutoModel
import torch
from demo_helpers.compute_performance import compute_performance
from demo_helpers.args import parse_args

from groqflow import groqit


def evaluate_minilm(rebuild_policy=None, should_execute=True):
    # set seed for consistency
    torch.manual_seed(0)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # load pre-trained torch model
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # dummy inputs to generate the groq model
    max_seq_length = 128
    dummy_inputs = {
        "input_ids": torch.ones((2, max_seq_length), dtype=torch.long),
        "token_type_ids": torch.ones((2, max_seq_length), dtype=torch.long),
        "attention_mask": torch.ones((2, max_seq_length), dtype=torch.bool),
    }

    # generate groq model
    groq_model = groqit(model, dummy_inputs, rebuild=rebuild_policy)

    # compute performance on CPU and GroqChip
    if should_execute:
        compute_performance(
            groq_model,
            model,
            dataset="stsb_multi_mt",
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            task="sentence_similarity",
        )

    print(f"Proof point {__file__} finished!")


if __name__ == "__main__":
    evaluate_minilm(**parse_args())
