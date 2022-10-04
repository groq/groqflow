# ResNet50

ResNet50 is a Convolutional Neural Network (CNN) model used for image classification. Kaiming He, et al. first introduced ResNet models and the revolutionary residual connection (also known as skip connection) in their 2015 paper, [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). The residual connection enables easier optimization and better accuracy while training deep models.

This proof point uses a [ResNet50 model](https://pytorch.org/hub/pytorch_vision_resnet/) pre-trained on the [ImageNet dataset](https://www.image-net.org/) and downloaded from PyTorch's model hub. The model is evaluated on the sampled, 10 class version of the ImageNet dataset, [Imagenette](https://github.com/fastai/imagenette).

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

To build and evaluate ResNet50:

  ```bash
  python resnet50.py
  ```

**Note:** The Proof Points directory [readme.md](../../README.md) details how to build and execute on two machines.

## Expected Results

It takes approximately 18 minutes for ResNet50 to build and about 3 minutes to evaluate the implementation accuracies. The script returns the accuracies for both the PyTorch implementation on a CPU and the Groq implementation using a single GroqCard™ accelerator.
