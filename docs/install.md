# GroqFlow™ Installation Guide

The following includes the instructions for installation of GroqFlow.

## Step 1. Confirm Versions
Make sure that your combination of GroqWare™ Suite version, OS version, and Python version are compatible. Our supported matrix of versions is:

| GroqWare Version |  OS Version  | Python Version |
|------------------|--------------|----------------|
| 0.10.0           | Ubuntu 22.04 | 3.10           |
| 0.10.0           | Rocky 8.4    | 3.8            |
| 0.9.3            | Ubuntu 22.04 | 3.8            |
| 0.9.3            | Rocky 8.4    | 3.8            |
| 0.9.3            | Ubuntu 18.04 | 3.8            |
| 0.9.2.1          | Ubuntu 22.04 | 3.10           |

## Step 2. Download and install the GroqWare Suite version >=0.9.2.1 
Create an account or log in on our [portal](https://support.groq.com) to see the [GroqWare Quick Start Guide](https://support.groq.com/#/downloads/view/groqware-qsg) for installation instructions.

If you have Groq hardware, install both the Groq Developer Tools Package (groq-devtools) and the Groq Runtime Package (groq-runtime) to both build and run models on hardware. 

If you do not have Groq hardware, install the Groq Developer Tools Package (groq-devtools) to build and compile models to estimate performance.

## Step 3. Create and activate a virtual Conda environment

Download, install, and create a Miniconda virtual environment by copying and pasting the following commands into your terminal:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda create -n groqflow python=$GF_PYTHON_VERSION
conda deactivate
conda activate groqflow
```

Where `$GF_PYTHON_VERSION` is the version of Python corresponding to your OS and GroqWare version in the [compatibility chart](#step-1-confirm-versions) above.

> _Note_: It is important to deactivate your base Conda environment when first setting up a new GroqFlow environment. This helps to prevent Conda from making unwanted changes in the PATHs of your environments.

## Step 4. Pip install GroqFlow
If you are developing a **Pytorch, ONNX,** or **Hummingbird** model, install the `groqflow` package into your virtual environment by copying and pasting the following commands into your terminal:

```
git clone https://github.com/groq/groqflow.git
pip install --upgrade pip
cd groqflow
pip install .
```

If you are developing a **Keras** model, use the following command instead of `pip install .`:

```
pip install .[tensorflow]
```

**Note:** Use separate Conda environments for PyTorch/ONNX/Hummingbird development vs. TensorFlow development. Do not `pip install groqflow[tensorflow]` into an environment where you already did `pip install groqflow`, as this will cause errors.

## Step 5. Add GroqWare Suite to Python Path

Copy and paste the following into your terminal to add GroqWare Suite to your PATH:

```
conda env config vars set PYTHONPATH="/opt/groq/runtime/site-packages:$PYTHONPATH"
```

**Note:** If you encounter errors later that say GroqFlow is unable to find a tool from GroqWare Suite (Groq API, Groq Runtime, Groq DevTools, Groq Compiler, etc.), it usually means either:
- You forgot to complete this step.
- Your GroqWare Suite installation failed and you should reinstall GroqWare Suite following all the required steps outlined in the [GroqWare Quick Start Guide](https://support.groq.com/#/downloads/view/groqware-qsg).

## Step 6. Reactivate your Conda environment
Copy and paste the following into your terminal for the PATH modification in the previous step to take effect:

```
conda deactivate
conda activate groqflow
```

## Step 7. Rock it 🚀 with `groqit()`

To confirm that you're setup correctly, navigate to the examples folder at `groqflow/examples/` and run the `hello_world.py` example that can be found in the `keras`, `onnx`, and `pytorch` folder depending on your preferred framework:

```
cd groqflow/examples/<framework>
python hello_world.py
```

## Step 8. (Recommended) Take off 🚀 with a Proof Point

Included in the `groqflow/proof_points` directory are various machine learning models and linear algebra workloads. To run these proof points, install `groqflow/demo_helpers` in your `groqflow` environment:

```
cd groqflow/demo_helpers/
pip install -e .
```

You can learn more about how to run proof points [here](https://github.com/groq/groqflow/tree/main/proof_points)!
