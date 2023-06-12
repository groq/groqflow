# GroqFlow™ Installation Guide

The following describes how to install GroqFlow. These instructions enable users to build models for Groq hardware, as well as execute those builds in systems that have GroqCard™ accelerators physically installed.

## Prerequisites

### Check your versions

- Ensure that you are using one of the following Linux distributions: Ubuntu 18.04, Ubuntu 22.04 or Rocky 8.4.
- Download and install the GroqWare™ Suite version >=0.9.2.1.
  - For more information, see the GroqWare Quick Start Guide at [support.groq.com](https://support.groq.com).
  - To compile your model for Groq hardware, GroqFlow requires the Groq Developer Tools Package (groq-devtools). To run your compiled model on hardware, GroqFlow requires the Groq Runtime Package (groq-runtime).

Make sure that your combination of GroqWare™ Suite version, OS version, and Python version are compatible. Our supported matrix of versions is:

| GroqWare  | OS           | Python Version |
|-----------|--------------|----------------|
| 0.9.2.1   | Ubuntu 22.04 | 3.10           |
| 0.9.3     | Ubuntu 18.04 | 3.8            |
| 0.9.3     | Ubuntu 22.04 | 3.8            |
| 0.9.3     | Rocky 8.4    | 3.8            |
| 0.10.0    | Ubuntu 22.04 | 3.10           |
| 0.10.0    | Rocky 8.4    | 3.8            |

### Install GroqWare

Download and install the GroqWare Suite version >=0.9.2.
- For more information, see the GroqWare Quick Start Guide at [support.groq.com](https://support.groq.com).
- To compile your model for Groq hardware, GroqFlow requires the Groq Developer Tools Package (groq-devtools). To run your compiled model on hardware, GroqFlow requires the Groq Runtime Package (groq-runtime).

## Trying out GroqFlow

If you want to try out GroqFlow by running the [examples](https://github.com/groq/groqflow/tree/main/examples) and [proof points](https://github.com/groq/groqflow/tree/main/proof_points), we recommend that you take the following steps. If you want to use GroqFlow with your own environment and model, we suggest skipping ahead to [Developing with GroqFlow](#developing-with-groqflow).

### Step 1: Create and activate a virtual environment

First, download, install, and create a Miniconda virtual environment.

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda create -n groqflow python=$GF_PYTHON_VERSION
conda deactivate
conda activate groqflow
```

Where `$GF_PYTHON_VERSION` is the version of Python corresponding to your OS and GroqWare version in the [compatibility chart](#check-your-versions) above.

> _Note_: it is important to deactivate your base conda environment when first setting up a new groqflow environment. This helps to prevent conda from making unwanted changes in the PATHs of your environments.

### Step 2: Pip install GroqFlow

Install the `groqflow` package into your virtual environment:

```
git clone https://github.com/groq/groqflow.git
pip install --upgrade pip
cd groqflow
pip install .
```

where `groqflow` is the directory where you cloned the GroqFlow repo in the [prerequisites](#prerequisites).

**Note:** On GroqNode™ systems you will may run into an installation error that suggests that you install with the `--user` flag. If you encounter this error, please try `pip install . --user`.

_Optional_: if you want to use GroqFlow with TensorFlow, use this install command instead of `pip install .`:

```
pip install .[tensorflow]
```

### Step 3: Add GroqWare Suite to Python Path

This adds the Groq tools to your path:

```
conda env config vars set PYTHONPATH="/opt/groq/runtime/site-packages:$PYTHONPATH"
```

**Note:** you will need to reactivate your conda environment for this to take effect.

**Note:** if you encounter errors later that say GroqFlow is unable to find a tool from the GroqWare suite (Groq API, Groq Runtime, Groq DevTools, Groq Compiler, etc.) it usually means either:
- You forgot to complete this step.
- Your GroqWare Suite installation failed and you should attempt to re-install the GroqWare Suite.

### Step 4: Identify your groqit() card

GroqFlow sets the GroqCard to be of type A1.4 by default. If you have a Legacy A1.1 GroqCard, run the following command before running on hardware so the multi-card workloads can properly bring up the connections between the cards:

```
export GROQFLOW_LEGACY_A11=True
```

### Step 5: Rock-It with groqit()

To confirm that you're setup correctly, navigate to the examples folder at `groqflow/examples/` and run the `hello_world.py` example that can be found in the `keras`, `onnx`, and `pytorch` folder depending on your preferred framework:

```
cd groqflow/examples/<framework>
python hello_world.py
```

### Step 6: Take-off with a Proof Point

Included in the directory: `groqflow/proof_points`, are multiple examples of various machine learning and linear algebra workloads. To run these proof points, the `groqflow/demo_helpers` must be installed in your groqflow environment.

```
cd groqflow/demo_helpers/
pip install -e .
```

Then you can learn about how to run proof points [here](https://github.com/groq/groqflow/tree/main/proof_points).

## Developing with GroqFlow

When you are ready to try out your own model with GroqFlow, we recommend taking the following steps:

1. Activate the conda virtual environment where you are able to run your model
1. Install the GroqFlow package from PyPI:
  - If you are developing a PyTorch, ONNX, or Hummingbird model, use `pip install groqflow`
  - If you are developing a Keras model, use `pip install groqflow[tensorflow]`
1. Follow steps 3 and 4 in [Testing Out GroqFlow](#testing-out-groqflow) to complete setup
1. Import `groqflow` into the script where you are running your model and call `groqit(model, inputs)` to build your model (see the [examples](https://github.com/groq/groqflow/tree/main/examples) to learn more about calling `groqit()`)

**Note:** The supported Python/OS combinations in [Check your Versions](#check-your-versions) apply here as well.

**Note:** We recommend using separate conda environments for PyTorch/ONNX/Hummingbird development vs. TensorFlow development. The reason we make TensorFlow support optional in GroqFlow is to help you avoid dependency conflicts between the TensorFlow package and the other Groq/GroqFlow dependencies. Do not `pip install groqflow[tensorflow]` into an environment where you already did `pip install groqflow`, as this will cause errors.