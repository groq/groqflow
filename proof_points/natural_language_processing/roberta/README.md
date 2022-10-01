# RoBERTa

[RoBERTa](https://arxiv.org/abs/1907.11692) is one of many derivatives of the [BERT model](https://arxiv.org/abs/1810.04805). Its name is an acronym created from the phrase, "Robustly optimized BERT approach". RoBERTa improves on BERT by hyperparameter tuning and altering the training recipe. Optimizations employed by RoBERTa include longer training with larger batch sizes, more data, longer sequence lengths, and dynamically changing masking patterns. As with many of the other BERT model variations, RoBERTa also removes the next sentence proposal (NSP) loss from the loss function.

In this proof point, RoBERTa is used for the task of [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition) and evaluated using the [CoNLL-2003 dataset](https://paperswithcode.com/dataset/conll-2003). The model weights are downloaded from the [Hugging Face website](https://huggingface.co/dominiqueblok/roberta-base-finetuned-ner).

## Prerequisites

- Ensure you've completed the install prerequisites:
  - Installed the GroqWare™ Suite
  - Installed GroqFlow
  - Installed Groq Demo Helpers
    - For more information on these steps, see the [Proof Points README](../../README.md).
- Install the python dependencies using the requirements.txt file included with this proof point using the following command:

  ```bash
  pip install -r requirements.txt
  ```

## Build and Evaluate

To build and evaluate RoBERTa:

  ```bash
  python roberta.py
  ```

**Note:** The Proof Points directory [readme.md](../../README.md) details how to build and execute on two machines.

## Expected Results

It takes approximately 15 minutes for RoBERTa to build and about 5 minutes to evaluate the implementation accuracies. The script returns the accuracies for both the PyTorch implementation on a CPU and the Groq implementation on 4 GroqCard™ accelerators within a GroqNode™ server.
