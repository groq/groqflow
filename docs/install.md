# GroqFlow™ Installation Guide

The following describes how to install GroqFlow. These instructions enable users to build models for Groq hardware, as well as execute those builds in systems that have GroqCard™ accelerators physically installed.

## Prerequisites

- Ensure that you are using one of the following Linux distributions: Ubuntu 18.04, Ubuntu 22.04 or Rocky 8.4.
- Download and install the GroqWare™ Suite version >=0.9.2.
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
conda create -n groqflow python=3.10
conda activate groqflow
```

### Step 2: Pip install GroqFlow

Install the `groqflow` package into your virtual environment:

```
pip install --upgrade pip
cd groqflow
pip install .
```

where `groqflow` is the directory where you cloned the GroqFlow repo in the [prerequisites](#prerequisites).

### Step 3: Add GroqWare Suite to Python Path

This adds the Groq tools to your path:

```
conda env config vars set PYTHONPATH="/opt/groq/runtime/site-packages:$PYTHONPATH"
```

**Note:** you will need to reactivate your conda environment for this to take effect.

### Step 4: Rock-It with groqit()

To confirm that you're setup correctly, navigate to the examples folder at `groqflow/examples/` and run the `hello_world.py` example that can be found in the `keras`, `onnx`, and `pytorch` folder depending on your preferred framework:

```
python hello_world.py
```

### Step 5: Take-off with a Proof Point

Included in the directory: `groqflow/proof_points`, are multiple examples of various machine learning and linear algebra workloads. To run these proof points, the `groqflow/demo_helpers` must be installed in your groqflow environment.

```
cd groqflow/demo_helpers/
pip install -e .
```

### Step 6: Identify your groqit() card

GroqFlow sets the GroqCard to be of type A1.4 by default. If you have a Legacy A1.1 GroqCard, run the following command before running on hardware so the multi-card workloads can properly bring up the connections between the cards:

```
export GROQFLOW_LEGACY_A11=True
```

