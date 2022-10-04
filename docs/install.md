# GroqFlow™ Installation Guide

The following describes how to install GroqFlow. These instructions enable users to build models for Groq hardware, as well as execute those builds in systems that have GroqCard™ accelerators physically installed.

## Prerequisites

- Download and install the GroqWare™ Suite version 0.9.0.
  - For more information, see the GroqWare Quick Start Guide at [support.groq.com](https://support.groq.com).
  - To compile your model for Groq hardware, GroqFlow requires the Groq Developer Tools Package (groq-devtools). To run your compiled model on hardware, GroqFlow requires the Groq Runtime Package (groq-runtime).
- Clone the GroqFlow GitHub repo using the following command:

```
git clone https://github.com/groq/groqflow.git
```

### Step 1: Create and activate a virtual environment

The following example demonstrates downloading, installing and creating a Miniconda virtual environment.

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda create -n groqflow python=3.8.13
conda activate groqflow
```

### Step 2: Pip install GroqFlow

Install the `groqflow` package into your virtual environment:

```
pip install --upgrade pip
cd groqflow
pip install -e .
```

where `groqflow` is the directory where you cloned the GroqFlow repo in the [prerequisites](#prerequisites).

### Step 3: Add GroqWare Suite to Python Path

This adds the Groq tools to your path:

```
conda install conda-build
conda develop /opt/groq/runtime/site-packages
```

### Step 4: Rock-It with groqit()

To confirm that you're setup correctly, navigate to the examples folder at `groqflow/examples/` and run the `hello_world.py` example that can be found in the `keras`, `onnx`, and `pytorch` folder depending on your preferred framework:

```
python hello_world.py
```
