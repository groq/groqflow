# SqueezeNet

[SqueezeNet](https://arxiv.org/abs/1602.07360?context=cs) is advertised as a small convolutional neural network (CNN) that achieves "AlexNet level accuracy on ImageNet with 50x fewer parameters" as quoted in the linked paper. SqueezeNet models are highly efficient in terms of size and speed while providing relatively good accuracies. This makes them ideal for platforms with strict constraints on size.

In this proof point, SqueezeNet is performing image classification. It is evaluated on the [Imagenette dataset](https://github.com/fastai/imagenette) which is a sampled, 10 class version of the [ImageNet dataset](https://www.image-net.org/). The model weights will be downloaded from the [PyTorch website](https://pytorch.org/hub/pytorch_vision_squeezenet/).

## Prerequisites

- Ensure you've completed the install prerequisites:
  - Installed the GroqWare™ Suite
  - Installed GroqFlow
  - Installed Groq Demo Helpers
    - For more information on these steps, see the [Proof Points README](../../README.md).
- Install the python dependencies for this proof point with the following:

  ```bash
  pip install -r requirements.txt
  ```

## Build and Evaluate

To build and evaluate SqueezeNet:

  ```bash
  python squeezenet.py
  ```

**Note:** The Proof Points directory [readme.md](../../README.md) details how to build and execute on two machines.

## Expected Results

It takes approximately 5 minutes for SqueezeNet to build and about 1 minute to evaluate the implementation accuracies. The script returns the accuracies for both the PyTorch implementation on a CPU and the Groq implementation using a single GroqCard™ accelerator.
