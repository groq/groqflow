# DistilBERT

[DistilBERT](https://arxiv.org/pdf/1910.01108.pdf) is a [distilled model](https://arxiv.org/pdf/1503.02531.pdf) using the [BERT model](https://arxiv.org/abs/1810.04805) as the teacher. DistilBERT has the same general architecture as BERT except half the layers, the pooler, and token embeddings are removed. This reduction in size allows the model to train faster and requires much less memory and power to run. DistilBert boasts that it retains 97% of the Bert model scores with 40% fewer parameters.

In this proof point, DistilBert performs the task of [Sentiment Classification](https://paperswithcode.com/task/sentiment-analysis) and is evaluated using the Stanford Sentiment Treebank [(SST) dataset](https://paperswithcode.com/dataset/sst). The model weights are downloaded from the [Hugging Face website](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).

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

To build and evaluate DistilBERT:

  ```bash
  python distilbert.py
  ```

**Note:** The Proof Points directory [readme.md](../../README.md) details how to build and execute on two machines.

## Expected Results

It takes approximately 8 minutes for DistilBERT to build and about 2 minutes to evaluate the model's accuracy. The example returns the accuracies for both the PyTorch implementation on a CPU and the Groq implementation on 2 GroqCard™ accelerators within a GroqNode™ server.
