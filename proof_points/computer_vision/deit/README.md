# DeiT

The [DeiT](https://arxiv.org/abs/2012.12877) (Data-efficient image Transformer) is a convolution-free transformer model designed for computer vision. DeiT models are efficiently trained for image classification using a novel token distillation process that can learn more from a convolutional teacher model than a transformer teacher. DeiT models also require less data to train than the original Vision Transformers [(ViT)](https://arxiv.org/abs/2010.11929v2).

This proof point obtains a [pre-trained DeiT-tiny](https://huggingface.co/facebook/deit-tiny-patch16-224) from Hugging Face for the task of Image Classification. The model implementations are evaluated using the 10-class [Imagenette dataset](https://github.com/fastai/imagenette) which is a sampling from the [ImageNet dataset](https://www.image-net.org/). The tiny version of the DeiT model illustrates the ability of GroqFlow™ and GroqWare™ Suite to support all of the necessary operations used to build and run the ConvNeXt models.

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

To build and evaluate DeiT-tiny:

  ```bash
  python deit_tiny.py
  ```

**Note:** The Proof Points directory [readme.md](../../README.md) details how to build and execute on two machines.

## Expected Results

It takes approximately 4 minutes for DeiT-tiny to build and about 2 minutes to evaluate the implementation accuracies. The script returns the accuracies for both the PyTorch implementation on a CPU and the Groq implementation on a GroqCard™ accelerator.
