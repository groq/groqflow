# YOLO v6

YOLOv6 is a Convolutional Neural Network (CNN) model used for [Object Detection](https://en.wikipedia.org/wiki/Object_detection). It is an extension of the original YOLO model developed by [Joseph Redmon](https://pjreddie.com/), et al. in their 2015 paper, [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640). The key innovation of YOLO is the improved efficiency in inference speed and computation compared to other object detection models. Yolo locates the objects in an image and classifies them in a single "look". Other state of the art image detection models use a many module approach which required separate steps to first identify possible objects and then another to classify located objects. Redmon argued that this required multiple "looks" at an image and while it could achieved good results, they were larger, more computationally intense, and therefore slower.

This variation of YOLO  was released by the Meituan Vision AI Department and [published on github](https://github.com/meituan/YOLOv6) in different sizes ranging from YOLOv6-nano at 4.3M parameters to YOLOv6-large at 58.5M parameters. This proof point compiles the YOLOv6-nano model for an input size of 640 X 640 pixels.

This proof point evaluates YOLOv6-nano on the [COCO dataset](https://cocodataset.org/). The success of the model is measured using the "mAP @ 0.5:0.95" metric, which computes an average mAP (Mean Average Precision) using different IoU (Intersection over Union) thresholds varying from 0.5 to 0.95. An explanation of this evaluation method can also be found at the COCO website under the [Evaluate tab](https://cocodataset.org/#detection-eval).

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

To build and evaluate YOLOv6-nano:

  ```bash
  python yolov6_nano.py
  ```

**Note:** The Proof Points directory [readme.md](../../README.md) details how to build and execute on two machines.

## Expected Results

It takes approximately 60 minutes for YOLOv6 to build and about 10 minutes to evaluate the implementation accuracies. The script returns the accuracies for both the PyTorch implementation on a CPU and the Groq implementation using a single GroqCard™ accelerator.
