# BERT

This folder contains proof points that demonstrate two variants of the Natural Language processing model, [BERT](https://arxiv.org/pdf/1810.04805.pdf): BERT-tiny, and BERT-base. BERT is a bidirectional transformer architecture pretrained using Masked Language Modeling. The success of these proof points illustrate the ability of GroqFlow and the GroqWare™ Suite to support both the operations and size of the classic transformer architecture used by BERT models.

BERT-tiny is a small (tiny, even!) variant of the BERT architecture. The paper, [Well-Read Students Learn Better: On the Importance of Pre-training Compact Models](https://arxiv.org/pdf/1908.08962.pdf), introduces BERT-tiny along with other BERT variants of reduced size: BERT-mini, BERT-small, and BERT-medium.  They are studied further in the paper [Generalization in NLI: Ways (Not) To Go Beyond Simple Heuristics](https://arxiv.org/pdf/2110.01518.pdf)

The Bert-tiny proof point uses a model fine-tuned on the [Stanford Sentiment Treebank (SST) dataset](https://paperswithcode.com/dataset/sst), loaded from [Huggingface](https://huggingface.co/M-FAC/bert-tiny-finetuned-sst2) to perform [Sentiment Classification](https://paperswithcode.com/task/sentiment-analysis).

The BERT-base proof point also uses a pre-trained model that is fine-tuned on the SST dataset for Sentiment Classification. [Huggingface](https://huggingface.co/howey/bert-base-uncased-sst2) provides the BERT-base model.

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

To build and evaluate BERT-tiny:

  ```bash
  python bert_tiny.py
  ```

To build and evaluate BERT-base:

  ```bash
  python bert_base.py
  ```

Note: The Proof Points directory [readme.md](../../README.md) details how to build and execute on two machines.

## Expected Results

 Each script returns the accuracies for both the PyTorch implementation on a CPU and the Groq implementation. The table below details the approximate time to run each part of the script, and the required number of GroqCard™ accelerator.

| Proof Point Model | Approx Build Time | Approx Evaluation Time | Num of GroqCard™ Accelerators |
|:-----------|:--------|:---------|:----------|
| BERT-tiny | 1 min | 30 sec | 1 |
| BERT-base | 15 min | 4 min | 4 |
