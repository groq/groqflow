# M5

[M5](https://arxiv.org/abs/1610.00087) is a convolutional neural network (CNN) that works directly on raw audio waveform. Since M5 accepts raw data, there is no need to generate frequency spectrums, a required pre-processing step used by many audio/acoustic models.

This proof point uses the M5 model on the task of [Keyword Spotting](https://en.wikipedia.org/wiki/Keyword_spotting). The M5 adaptation for this task replaces the global average pool in the original M5 model with a fully connected layer; the architecture definition can be viewed in the [demo_helpers folder](../../../demo_helpers/models.py).

M5's Keyword Spotting accuracy is evaluated using the [SpeechCommands dataset](https://arxiv.org/abs/1804.03209) from PyTorch's `torchaudio.datasets` library.

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

- Since this proofpoint uses audio files, often the audio libraries must be installed on system.
  - For Ubuntu OS:

  ```bash
  sudo apt install libsox-dev
  ```

  - For Rocky OS:

  ```bash
  sudo dnf install sox-devel
  ```

## Build and Evaluate

To build and evaluate M5:

  ```bash
  python m5.py
  ```

Note: The [Proof Points directory README](../../README.md) details how to build and execute on two machines.

## Expected Results

It takes approximately 5 minutes for M5 to build and about 1 minute to evaluate the implementation accuracies. The script returns the accuracies for both the PyTorch implementation on a CPU and the Groq implementation using a single GroqCard™ accelerator.
