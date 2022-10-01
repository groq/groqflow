# MiniLM v2

[MiniLM v2](https://arxiv.org/abs/2012.15828) is a [distilled model](https://arxiv.org/pdf/1503.02531.pdf) that employs a generalization of the deep self-attention distillation method that the authors of the linked paper introduced in their first paper [MiniLm](https://arxiv.org/abs/2002.10957). The distillation is generalized by employing multi-head self-attention distillation.

In this proof point, MiniLM v2 is used for the task of [sentence similarity](https://huggingface.co/tasks/sentence-similarity) and evaluated using the [machine translated multi-lingual](https://github.com/PhilipMay/stsb-multi-mt) version of the Semantic Textual Similarity [(STS) benchmark dataset](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark). Both the [model](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) and the [dataset](https://huggingface.co/datasets/stsb_multi_mt#citation-information) are downloaded from Hugging Face.

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

To build and evaluate MiniLM v2:

  ```bash
  python minilmv2.py
  ```

**Note:** The Proof Points directory [readme.md](../../README.md) details how to build and execute on two machines.

## Expected Results

It takes approximately 10 minutes for MiniLM v2 to build and about 1 minutes to evaluate the [Spearman Rank Correlation Coefficient](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) for both implementations. The script returns the Spearman Rank Correlation Coefficients for both the PyTorch implementation on a CPU and the Groq implementation using a single GroqCard™ accelerator.
