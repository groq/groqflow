# ELECTRA

[ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB) uses a self-supervised pre-training method for language representation learning that is similar to a [Generative Adversarial Network (GAN)](https://en.wikipedia.org/wiki/Generative_adversarial_network), without the adversarial part. During pre-training, instead of masking an input token and learning what the masked token is, like many other NLP models, a few input tokens are replaced with tokens of similar meaning by a small generative network. Then, the edited input is fed into a discriminator network to learn to differentiate between the original and replacement tokens. After training, the generator network is discarded and the discriminator network is used for inference. With this architecture and training method, ELECTRA boasts that it learns more efficiently and meets or outperforms, in terms of accuracy, models that only learn the masked tokens.

In this proof point, ELECTRA is fine-tuned on the [Stanford Sentiment Treebank (SST) dataset](https://paperswithcode.com/dataset/sst), loaded from [Huggingface](https://huggingface.co/M-FAC/bert-tiny-finetuned-sst2), and performs the task of [Sentiment Classification](https://paperswithcode.com/task/sentiment-analysis).

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

To build and evaluate ELECTRA:

  ```bash
  python electra.py
  ```

**Note:** The Proof Points directory [readme.md](../../README.md) details how to build and execute on two machines.

## Expected Results

It takes approximately 15 minutes for ELECTRA to build and about 4 minutes to evaluate the implementation accuracies. The script returns the accuracies for both the PyTorch implementation on a CPU and the Groq implementation on 4 GroqCard™ accelerators within a GroqNode™ server.
