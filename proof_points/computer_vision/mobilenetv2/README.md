# MobileNetV2

[MobileNetV2](https://arxiv.org/abs/1801.04381) is a CNN model that was designed to perform well on mobile devices. The architecture makes use of an inverted residual structure where the residual connections are between the bottleneck layers. The intermediate expansion layer uses lightweight depthwise convolutions to filter features as a source of non-linearity and to reduce the memory footprint of the model.

This proof point uses a [MobileNet V2 model]((https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)) pre-trained on the [ImageNet dataset](https://www.image-net.org/) and downloaded from PyTorch's model hub. The model is evaluated on the sampled, 10 class version of the ImageNet dataset, [Imagenette](https://github.com/fastai/imagenette).

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

# Build and Evaluate

To build and evaluate MobileNetV2:

  ```bash
  python mobilenetv2.py
  ```

**Note:** The Proof Points directory [readme.md](../../README.md) details how to build and execute on two machines.

## Expected Results

It takes approximately 12 minutes for MobileNetV2 to build and about 2 minutes to evaluate the implementation accuracies. The script returns the accuracies for both the PyTorch implementation on a CPU and the Groq implementation using a single GroqCard™ accelerator.
