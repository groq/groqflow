# GroqFlow™ Examples

This folder contains examples that demonstrate the use of `groqit()` arguments and `GroqModel` methods.

You can learn more about the concepts demonstrated in the examples by referencing the GroqFlow User Guide at `docs/user_guide.md`.

## Table Of Contents

- [Groq Tool Requirements](#groq-tool-requirements)
- [Understanding Examples](#understanding-examples)
- [Running Examples](#running-examples)
- [Hello Worlds](#hello-worlds)
- [Hummingbird Examples](#hummingbird-examples)
- [Additional Pytorch Examples](#additional-pytorch-examples)

## Groq Tool Requirements

The Groq tools packages and the **Quick Start Guide** can be found at the [Groq Customer Portal](https://support.groq.com/)

- To build a `groq_model` the `groq-devtools` package should be installed.
- To run a `GroqModel` on hardware the `groq-runtime` package should be installed.
- Both Groq packages should be installed to enable both a build and to run on hardware
  from the same script.

## Understanding Examples

Here are some properties shared by all of the examples:

- Each example will create a build directory in the GroqFlow build cache, which is located at `~/.cache/groqflow` by default.
  - **Note**: Most builds will load from this cache after the first time you run them, as opposed to rebuilding, unless otherwise specified in the example (check out the `rebuild` argument and its examples to change this behavior).
  - **Note**: Most examples set `torch.manual_seed(0)` or `tf.random.set_seed(0)`, unless otherwise specified in the example, which prevents the randomly generated weights in the example from changing between runs.
- The build directory will be named after the example unless the example specifies a name change with the `build_name` argument (see the `build_name.py` example).
- The model being built in each example is a small one- or two-layer fully-connected graph.

## Running Examples

To run any of the examples, open a terminal and type the following command:

```python
python /path/to/example/example_name.py
```

## Hello Worlds

| **Example Name** | **Demonstrates** |
|:--------|:-----------|
| `pytorch/hello_world.py` | building and running a model defined in PyTorch|
| `keras/hello_world.py` | building and running a model defined in Keras|
| `onnx/hello_world.py` | building and running a model defined as an ONNX file|

## Hummingbird Examples

| **Example Name** | **Demonstrates** |
|:--------|:-----------|
| `hummingbird/randomforest.py` | building and running a Hummingbird RandomForestClassifier against random data |
| `hummingbird/xgbclassifier.py` | building and running a Hummingbird XGBClassifier against random data |

## Additional PyTorch Examples

| **Example Name** | **Demonstrates** |
|:--------|:-----------|
| `pytorch/assembler_flags.py` | the `assembler_flags` argument to `groqit()` |
| `pytorch/benchmark.py` | the `benchmark()` method of `GroqModel` |
| `pytorch/benchmark_abunch.py` | the `benchmark_abunch()` method of `GroqModel` |
| `pytorch/build_name.py` | the `build_name` argument to `groqit()` |
| `pytorch/cache_dir.py` | the `cache_dir` argument to `groqit()` |
| `pytorch/compiler_flags.py` | the `compiler_flags` argument to `groqit()` |
| `pytorch/estimate_performance.py` | the performance estimation feature of GroqFlow |
| `pytorch/groqview.py` | how to create and open a GroqView visualization using GroqFlow |
| `pytorch/no_monitor.py` | the `monitor` argument to `groqit()` |
| `pytorch/num_chips.py` | the `num_chips` argument to groqit()|
| `pytorch/rebuild_always.py` | `groqit()`'s caching behavior when the `rebuild` argument is set to "always" |
| `pytorch/rebuild_never.py` | groqit()'s caching behavior when the `rebuild` argument is set to "never" |
| `pytorch/run_abunch.py` | running multiple inputs at a time with the `run_abunch()` method |
