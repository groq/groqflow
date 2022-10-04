# GroqFlow Proof Points

Proof points demonstrate how GroqFlow is able to successfully build and execute a model on Groq hardware, while maintaining model accuracy. The models are organized by category.

- Computer Vision (CV)
- Natural Language Processing (NLP)
- Speech Processing

## Table of Contents

- [Prerequisites](#prerequisites)
- [Support Matrix](#support-matrix)
  - [Computer Vision](#computer-vision)
  - [Natural Language Processing](#natural-language-processing)
  - [Speach Processing](#speech-processing)
- [Running a Script](#running-a-script)
- [Build and Evaluate on a Single Machine](#build-and-evaluate-on-a-single-machine)
- [Build and Evaluate on Separate Machines](#build-and-evaluate-on-separate-machines)

## Prerequisites

The following tasks are required to enable running proof point scripts:

- Download and install the GroqWare Suite packages from from the [Groq Customer Portal](https://support.groq.com/):
  - `groq-devtools` package, for model development and builds
  - `groq-runtime` package, for running computations on hardware (Groq Hardware must be present to install)
  - If building and executing a proof point on the same host machine, download and install both of the above packages.
- Clone the [GroqFlow Repository](https://github.com/groq/groqflow)
- Set up and activate a `groqflow` environment
  - Follow the [GroqFlow Installation Guide](https://github.com/groq/groqflow/blob/main/docs/install.md)
- Pip install the helper files for the proof points
  - `pip install -e {path_to}/groqflow/demo_helpers`

## Support Matrix

The following relates the proof point models with the version of the GroqWare Suite (SDK) in which they are supported.

### Computer Vision

| Proof Point Model | Supported SDK Version(s)|
|:------------------|:------------------------|
| [DeiT-tiny](computer_vision/deit/) | 0.9.0
| [GoogleNet](computer_vision/googlenet/) | 0.9.0
| [MobileNetV2](computer_vision/mobilenetv2/) | 0.9.0
| [ResNet50](computer_vision/resnet50/) | 0.9.0
| [SqueezeNet](computer_vision/squeezenet/) | 0.9.0

### Natural Language Processing

| Proof Point Model | Supported SDK Version(s)|
|:------------------|:------------------------|
| [Bert Tiny](natural_language_processing/bert/) | 0.9.0
| [Bert Base](natural_language_processing/bert/) | 0.9.0
| [DistilBERT](natural_language_processing/distilbert/) | 0.9.0
| [ELECTRA](natural_language_processing/electra/) | 0.9.0
| [MiniLM v2](natural_language_processing/minilm/) | 0.9.0
| [RoBERTa](natural_language_processing/roberta/) | 0.9.0

### Speech Processing

| Proof Point Model | Supported SDK Version(s)|
|:------------------|:------------------------|
| [M5](speech/m5/) | 0.9.0

## Running A Script

Each proof point will first build a GroqModel, and then evaluate the model on both a CPU and Groq hardware. If access to Groq hardware is available, the build and model evaluation steps can be run in a single step. However, a two step process has also been provided in case resource management requires that the build and evaluation steps be carried out on separate machines. Provided here are the general steps to run a script, but each proof point has a README with that provides any requirements and features that are specific to the model.

**Note**: Builds for large models can take several minutes. To avoid a time commitment surprise, the build time is included in the README for each proof point.

## Build and Evaluate on a Single Machine

Navigate to the folder containing the proof point and read the model's details in the `README`.

- Install the `requirement.txt` file.

  ```bash
  pip install -r requirements.txt
  ```

- Build and evaluate the proof point:

  ```bash
  python {proof_point_name}.py
  ```

## Build and Evaluate on Separate Machines

Navigate to the folder containing the proof point and read the model's details in the `README`.

- Install the `requirement.txt` file.

  ```bash
  pip install -r requirements.txt
  ```

- Build the model by running the command with the `--build` flag as shown below:

  ```bash
  python {proof_point_name}.py --build
  ```

  - If the model already exists in cache, it will not be rebuilt unless the model code or build changes.
- Transfer the proof point script and the `.iop` files to the machine connected to Groq Hardware.
  - The resulting build artifacts will be located in the GroqFlow Cache directory for the proof point, `~/.cache/groqflow/{proof_point_name}`. These artifacts include log files, ONNX files, inputs, the yaml state file, and the compile folder.
  - The `.iop` files can be found within the compile folder in the cache directory. There will be a file for each card used to execute the model on Groq hardware. Copy these files to the same location on the second machine:

    `~/.cache/groqflow/{proof_point_name}/compile/*.iop`

- Once the proof point is copied to the same cache directory and the initial prerequisites are met, the script can be run a second time with the `--execute` flag to evaluate the model.

  ```bash
  python {proof_point_name}.py --execute
  ```
