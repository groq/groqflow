# GroqFlow™ User Guide

The following reviews the different functionality provided by GroqFlow.

## Table of Contents

- [Just Groq It](#just-groq-it)
- [`groqit()` Arguments](#groqit-arguments)
  - [Quickest Way](#quickest-way)
  - [Multi-Chip](#multi-chip)
  - [Turn off the Progress Monitor](#turn-off-the-progress-monitor)
  - [Rebuild Policy](#rebuild-policy)
  - [Set the Build Name](#setting-the-build-name)
  - [Build a GroqView™ Visualization](#build-a-groqview™-visualization)
  - [Compiler Flags](#compiler-flags)
  - [Assembler Flags](#assembler-flags)
  - [Choose a Cache Directory](#choose-a-cache-directory)
- [GroqModel Methods](#groqmodel-methods)
  - [GroqModel Class](#groqmodel-class)
  - [GroqModel Specializations](#groqmodel-specializations)
  - [Calling an Inference](#inference-forward-pass)
  - [Benchmarking the Model](#benchmark)
  - [Netron](#netron)
  - [Open a GroqView Visualization](#open-a-groqview-visualization)
- [Concepts](#concepts)
  - [GroqFlow Build Cache](#groqflow-build-cache)
  - [`state.yaml` File](#stateyaml-file)


## Just Groq It

```
from groqflow import groqit     # import our function
gmodel = groqit(model, inputs)  # returns a callable GroqModel
gmodel(**inputs)                # inference with provided inputs
```

---


## `groqit()` Arguments

### Quickest Way

The simplest way to use GroqFlow is by calling `groqit()` with your model and a sample input.

Returns a callable `GroqModel` instance that works like a PyTorch model (torch.nn.Module).

**model:**

- Model to be mapped to Groq hardware.
- Can be an instance of:
  - PyTorch model (torch.nn.Module)
  - Keras model (tf.keras.Model)
  - TorchScript model (torch.jit.ScriptModule)
  - ONNX model (path/to/.onnx)

**inputs:**

- Used by `groqit()` to determine the shape of input to build against.
- Dictates the maximum input size the model will support.
- Same exact format as your model inputs.
- Inputs provided here can be dummy inputs.
- *Hint*: At runtime, pad your inference inputs to this expected input size.

> Good: allows for an input length of up to 128

`inputs = tokenizer("I like dogs", padding="max_length", max_length=128)`

> Bad: allows for inputs only the size of "I like dogs" and smaller

`inputs = tokenizer("I like dogs")`

### Examples:

```
groqit(my_model, inputs)
```

See
 - `examples/pytorch/hello_world.py`
 - `examples/keras/hello_world.py`

---

### Multi-Chip

By default, GroqFlow will automatically partition models across multiple GroqChip™ processors, however, a user can still specify the desired number of GroqChip™ processors they would like `groqit()` to target.

**num_chips**

- Number of GroqChip processors to be used.
- *Default*: `groqit()` automatically selects a number of chips.
- 1, 2, 4, or 8 chips are valid options for systems using GroqCard™ accelerators (GC1-010B/GC1-0109).

### Example:

```
groqit(model, inputs, num_chips=4)
```

See: `examples/num_chips.py`

---

### Turn off the Progress Monitor

GroqFlow displays a monitor on the command line that updates the progress of `groqit()` as it builds. By default, this monitor is on, however, it can be disabled using the `monitor` flag.

**monitor**
- *Default*: `groqit(monitor=True, ...)` displays a progress monitor on the command line.
- Set `groqit(monitor=False, ...)` to disable the command line monitor.

### Example:

```
groqit(model, inputs, monitor=False)
```

See: `examples/pytorch/no_monitor.py`

---

### Rebuild Policy

By default, GroqFlow will load successfully built models from the [GroqFlow build cache](#groqflow-build-cache) if they are available. Builds that have become stale will also be rebuilt to ensure correctness.

However, sometimes you may want to change this policy. The `rebuild` argument has a few settings that allow you to do just that.

**rebuild**
- *Default*: `groqit(rebuild="if_needed", ...)` will use a cached model if available, build one if it is not available, and rebuild any stale builds.
- Set `groqit(rebuild="always", ...)` to force `groqit()` to always rebuild your model, regardless of whether it is available in the cache or not.
- Set `groqit(rebuild="never", ...)` to make sure `groqit()` never rebuilds your model, even if it is stale. `groqit()` will attempt to load any previously built model in the cache, however there is no guarantee it will be functional or correct.

### Example:

```
# Rebuild a model every time
groqit(model, inputs, rebuild="always")

# Never rebuild a model
groqit(model, inputs, rebuild="never")
```

See:
 - `examples/pytorch/rebuild_always.py`
 - `examples/pytorch/rebuild_never.py`

---

### Setting the Build Name

By default, `groqit()` will use the name of your script as the name for your build in the [GroqFlow build cache](#groqflow-build-cache). For example, if your script is named `my_model.py`, the default build name will be `my_model`.

However, you can also specify the name using the `build_name` argument.

If you want to build multiple models in the same script, you must set a unique `build_name` for each to avoid collisions.


**build_name**
- Name of the build in the [GroqFlow build cache](#groqflow-build-cache), specified by `groqit(build_name="name", ...)`

### Examples:

> Good: each build has its own entry in the [GroqFlow build cache](#groqflow-build-cache) and `gmodel_a` and `gmodel_b` will correspond to `model_a` and `model_b`, respectively.

```
gmodel_a = groqit(model_a, inputs_a, build_name="model_a")
gmodel_b = groqit(model_b, inputs_b, build_name="model_b")
```

> Bad: the two builds will collide, and the behavior will depend on your [rebuild policy](#rebuild-policy)

- `rebuild="if_needed"` and `rebuild="always"` will replace the contents of `gmodel_a` with `gmodel_b` in the cache.
  - `rebuild="if_needed"` will also print a warning when this happens.
- `rebuild="never"` will load `gmodel_a` from cache and use it to populate `gmodel_b`, and print a warning.

```
gmodel_a = groqit(model_a, inputs_a)
gmodel_b = groqit(model_b, inputs_b)
```

See: `examples/pytorch/build_names.py`

---

### Build a GroqView™ Visualization

GroqView is a visualization and profiler tool that is launched in your web browser. For more information about GroqView, see the GroqView User Guide on Groq's Customer Portal at [support.groq.com](https://support.groq.com/#/downloads/groqview-ug)


**groqview**
- By default, GroqView files are not included in the build because they increase the build time and take up space on disk.
- When calling `groqit()`, set the groqview argument to True such as, `groqit(groqview=True, ...)` to include GroqView files in the build.
- To open the visualization, take the resulting `GroqModel` instance and call the `GroqModel.groqview()` method.

### Examples:

```
gmodel = groqit(model, inputs, groqview=True)
gmodel.groqview()
```

See: `examples/pytorch/groqview.py`

---

### Compiler Flags

Users familiar with the underlying compiler may want to override the default flags that `groqit()` provides to Groq Compiler. For more information about the available compiler flags, see the Compiler User Guide on Groq's Customer Portal at [support.groq.com](https://support.groq.com/#/downloads/groqcompiler-ug)

Warning: at this time, `groqit()` does nothing to ensure that you are providing legal flags to the `compiler_flags` argument. If you provide illegal flags, `groqit()` will raise a generic exception and point you to a log file where you can learn more.

**compiler_flags**
- Provide the flags as a list of strings, i.e., `groqit(compiler_flags=["flag 1", "flag 2"], ...)`
  - *Note*: By providing flags, this overwrites the defaults flags used by GroqFlow.

### Example:

```
gmodel = groqit(model, inputs, compiler_flags = ['--disableAddressCompaction', '--extraInstMemSlices=12'])
```

See: `examples/pytorch/compiler_flags.py`

---

### Assembler Flags

Users familiar with the underlying assembler may want to override the default flags that `groqit()` provides to Groq Assembler. For more information about the available assembler flags, see the Compiler User Guide on Groq's Customer Portal at [support.groq.com](https://support.groq.com/#/downloads/groqcompiler-ug)

Warning: at this time, `groqit()` does nothing to ensure that you are providing legal flags to the `assembler_flags` argument. If you provide illegal flags, `groqit()` will raise a generic exception and point you to a log file where you can learn more.

**assembler_flags**
- Provide the flags as a list of strings, i.e., `groqit(assembler_flags=["flag 1", "flag 2"], ...)`
  - *Note*: By providing flags, this overwrites the defaults flags used by GroqFlow.

### Example:

```
gmodel = groqit(model, inputs, assembler_flags = ['--auto-agt-dim=2'])
```

See: `examples/pytorch/assembler_flags.py`

---

### Choose a Cache Directory

The location of the [GroqFlow build cache](#groqflow-build-cache) defaults to `~/.cache/groqflow`. However, there are two ways for you to customize this.

- On a per-build basis, you can set `groqit(cache_dir="path", ...)` to specify a path to use as the cache directory.
- To change the global default, set the `GROQFLOW_CACHE_DIR` environment variable to a path of your choosing.

### Example:

`cache_dir` argument:

```
gmodel = groqit(model, inputs, cache_dir="local_cache") # cache is created in the current working directory

>>> python my_model.py

Contents of the current working directory:
  my_model.py
  local_cache/
    my_model/
      ...
  ...
```

`GROQFLOW_CACHE_DIR` environment variable:

```
>>> export GROQFLOW_CACHE_DIR=~/groqflow_cache
>>> python my_model.py

Resulting contents of ~:
  groqflow_cache/
    my_model/
      ...
```

See: `examples/pytorch/cache_dir.py`

---

## GroqModel Methods

### GroqModel Class

Successful `groqit()` calls will return a functioning instance of `class GroqModel`, which implements your model for Groq hardware.

`GroqModel` is a base class that implements the `run()`, `run_abunch()`, `estimate_performance()`, `groqview()`, and `netron()` methods documented below. `groqit()` will also return specialized wrappers for `GroqModel` on a per-model-framework basis (see [GroqModel Specializations](#groqmodel-specializations)).

*Warning*: today's implementation of `GroqModel` is not meant to deliver performance in any meaningful way. Use `GroqModel` only for assessing the functionality of your GroqFlow builds.

The following assumes that you've previously included something similar to the following to obtain a `GroqModel` instance, `gmodel`:

```
gmodel = groqit(pytorch_model, inputs)
```

---

### GroqModel Specializations

ONNX models built with `groqit()` return a standard `GroqModel` instance.
 - Tensors returned by `GroqModel` are of type `np.array`

PyTorch and TorchScript models built with `groqit()` return an instance of `class PytorchModelWrapper(GroqModel)`, which is the same as a `GroqModel` except:
 - `PytorchModelWrapper` is a callable object, similar to `torch.nn.Module` (i.e., `__call__()` executes the model's forward function)
 - Tensors returned by `PytorchModelWrapper` will be of type `torch.tensor`

Keras models built with `groqit()` return an instance of `class KerasModelWrapper(GroqModel)`, which is the same as a `GroqModel` except:
 - `KerasModelWrapper` is a callable object, similar to `tf.keras.Model` (i.e., `__call__()` executes the model's call function)
 - Tensors returned by `KerasModelWrapper` will be of type `tf.Tensor`

---

### Inference (Forward Pass)

- Performing inference doesn't require rebuilding (the `GroqModel` will load from the [GroqFlow build cache](#groqflow-build-cache))
- *Hint*: Pad your inputs to the same shape used when creating the model

**`GroqModel.run(inputs: Dict)`**
- Performs an inference of your model against the `inputs` provided
- The `inputs` argument is a dictionary where the keys correspond to the arguments of your model's forward function

**`PytorchModelWrapper.__call__(**kwargs)`**
- A `GroqModel` based on a PyTorch model is callable like a PyTorch model
- The `inputs` argument is an unpacked dictionary where the keys correspond to the arguments of your model's forward function

### Example:

```
>>> pytorch_model(**inputs)
  tensor([0.245, 0.235, 0.235, 0.267])

>>> gmodel(**inputs)
  tensor([0.245, 0.235, 0.235, 0.267])

>>> gmodel.run(inputs)
  tensor([0.245, 0.235, 0.235, 0.267])

```

See:
 - `examples/pytorch/hello_world.py`
 - `examples/keras/hello_world.py`
 - `examples/onnx/hello_world.py`


---

### Multiple Inferences

`GroqModel` provides a method, `GroqModel.run_abunch()`, to help you run a bunch of inferences. Specifically, `run_abunch()` will iterate over a collection of inputs and run each one of them.

- The argument to `run_abunch(input_collection=...)` is a list of dictionaries, where each dictionary has keys corresponding to the arguments to your model's forward function.
- *Hint*: use `run_abunch()` when the overhead of using `GroqModel.__call__()` would make it onerous to evaluate a large dataset (see our [warning about the performance](#groqmodel-methods) of `GroqModel`)

### Example:

```
>>> for inputs in input_collection:
>>>     pytorch_model(**inputs)
  tensor([0.245, 0.235, 0.235, 0.267])
  tensor([0.345, 0.335, 0.335, 0.367])
  tensor([0.445, 0.435, 0.435, 0.467])


>>> gmodel.run_abunch(input_collection)
  tensor([0.245, 0.235, 0.235, 0.267])
  tensor([0.345, 0.335, 0.335, 0.367])
  tensor([0.445, 0.435, 0.435, 0.467])


```

See: `examples/pytorch/run_abunch.py`

---

### Performance Estimation

`GroqModel` provides a method, `GroqModel.estimate_performance()`, to help you understand the throughput and latency of your build. We implemented this method because `GroqModel` is not yet optimized for your performance.

`estimate_performance()` estimates performance as the sum of:
 - The exact (deterministic) amount of time it will take for GroqChip processors to perform the computation for your model.
 - An estimate of the amount of time the PCIe bus will take to transfer your inputs from CPU memory to GroqChip processor and retrieve your results back in CPU memory.

`estimate_performance()` returns a `GroqEstimatedPerformance`, which has members for:
 - `latency` in seconds.
 - `throughput` in inferences per second (IPS).

### Example:

```
>>> gmodel.estimate_performance().latency
  0.001

>>> gmodel.estimate_performance().throughput
  1000
```

See: `examples/pytorch/estimate_performance.py`

---

### Netron

- Opens a visualization of your model by passing the ONNX model generated by GroqFlow into Netron (third party software)
- Prerequisite: An installation of Netron

### Example:

```
gmodel = groqit(model, inputs)
gmodel.netron()
```

See: `examples/pytorch/netron.py`

---

### Open a GroqView Visualization

Use a `GroqModel` instance to open a GroqView visualization that was built using the `groqview` argument to `groqit()` (more information [here](#groqview)).

- Visualize data streams and execution schedule.
- Requires using the groqview argument to groqit() at build time.

```
gmodel = groqit(model, inputs, groqview=True)
gmodel.groqview()
```

See: `examples/pytorch/groqview.py`

## Concepts

### GroqFlow Build Cache

The *GroqFlow build cache* is a location on disk that holds all of the artifacts from your GroqFlow builds. The default cache location is `~/.cache/groqflow` (see [Choose a Cache Directory](#choose-a-cache-directory) for more information).
- Each build gets its own directory, named according to the `build_name` [argument](#setting-the-build-name), in the cache.
- A build is considered stale (will not be loaded by default) under the following conditions:
    - The model, inputs, or arguments to `groqit()` have changed since the last successful build.
        - Note: a common cause of builds becoming stale is when `torch` or `keras` assigns random values to parameters or inputs. You can prevent the random values from changing by using `torch.manual_seed(0)` or `tf.random.set_seed(0)`.
    - The major or minor version number of GroqFlow has changed, indicating a breaking change to builds.
- The artifacts produced in each build include:
    - Build information is stored in `*_state.yaml`, where `*` is the build's name (see [The state.yaml File](#the-stateyaml-file)).
    - Log files produced by each stage of the build (`log_*.txt`, where `*` is the name of the stage).
    - ONNX files (.onnx) produced by build stages,
    - Input/output program files (.iop) used for programming GroqChip processors.
    - etc.

### `state.yaml` File

The *state.yaml* file is a build artifact generated by `groqit()` that contains information about the state of the build. This file can be useful for your overall understanding of the build, as well as debugging failed builds.

Some interesting fields in state.yaml include:
 - `build_status`: this will say `successful_build` if the build successfully produced the artifacts required to instantiate a working `GroqModel`
 - `config`: the user's arguments to `groqit()` that can impact the resulting build
 - `groqflow_version`: version number of the GroqFlow package used for the build
 - `num_chips_used`: number of GroqChip processors required to run the build
 - etc.
