# BERT-tiny

BERT-tiny ([link1](https://arxiv.org/pdf/1908.08962.pdf), [link2](https://arxiv.org/pdf/2110.01518.pdf)) is a small (tiny, even!) variant of the pre-trained BERT architecture. Other small variants include BERT-mini, BERT-small and BERT-medium. The success of this BERT variant proof point illustrates the ability of GroqFlow™ and the GroqWare™ Suite to support all of the necessary operations used to build and run any [BERT model architecture](https://arxiv.org/abs/1810.04805).

In this proof point a pre-trained BERT-tiny model, fine-tuned on the [Stanford Sentiment Treebank (SST) dataset](https://paperswithcode.com/dataset/sst), is loaded from [Huggingface](https://huggingface.co/M-FAC/bert-tiny-finetuned-sst2) and performs the task of [Sentiment Classification](https://paperswithcode.com/task/sentiment-analysis).

## Prerequisites

- Ensure you've completed the install prerequisites:
  - Installed GroqWare™ Suite
  - Installed GroqFlow
  - Installed Groq Demo Helpers
    - For more information on these steps, see the [Proof Points README](../../README.md).
- Install the python dependencies using the requirements.txt file included with this proof point using the following command:

  ```bash
  pip install -r requirements.txt
  ```

## Build and Evaluate

To build and evaluate Bert-tiny:

  ```bash
  python bert_tiny.py
  ```

Note: The Proof Points directory [readme.md](../../README.md) details how to build and execute on two machines.

## Expected Results

It takes approximately 1 minute for Bert-tiny to build and about 30 seconds to evaluate the implementation accuracies. The script returns the accuracies for both the PyTorch implementation on a CPU and the Groq implementation using a single GroqCard™ accelerator.
