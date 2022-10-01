# GoogLeNet

[GoogLeNet](https://arxiv.org/abs/1409.4842v1) is the convolutional neural network (CNN) based on the Inception architecture that received top marks in the ImageNet Large-Scale Visual Recognition Challenge 2014 ([ILSVRC 2014](https://www.image-net.org/challenges/LSVRC/2014/)). The stacked Inception modules applied multiple convolutional filter sizes (1x1, 3x3, & 5x5) before aggregating the results so that the next stage could simultaneously extract features of varying scale. The number of parameters and computational complexity were kept in check by using 1x1 convolution layers before the larger filters to reduce the layer dimensions before convolving over large patch sizes.

In this proof point, GoogleNet is used for the task of image classification and evaluated using the Imagenette [dataset](https://github.com/fastai/imagenette), a 10 class, sampled version of the ImageNet [dataset](https://www.image-net.org/). The model weights are downloaded from the [PyTorch website](https://pytorch.org/hub/pytorch_vision_googlenet/).

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

To build and evaluate GoogLeNet:

  ```bash
  python googlenet.py
  ```

**Note:** The Proof Points directory [readme.md](../../README.md) details how to build and execute on two machines.

## Expected Results

It takes approximately 6 minutes for GoogLeNet to build and about 2 minutes to evaluate the implementation accuracies. The script returns the accuracies for both the PyTorch implementation on a CPU and the Groq implementation on a GroqCard™ accelerator.
