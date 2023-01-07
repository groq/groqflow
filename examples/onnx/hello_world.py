"""
    Hello ** ONNX ** World!

    This example uses a small model to carry out a single vector matrix
    multiplication to demonstrate building and running an ONNX model
    with GroqFlow.

    This example will help identify what you should expect from each groqit()
    ONNX build. You can find the build results in the cache directory at
    ~/.cache/groqflow/hello_onnx_world/ (unless otherwise specified).
"""

import os
import torch
from groqflow import groqit
import onnxruntime as ort

torch.manual_seed(0)

# Start from a PyTorch model so you can generate an ONNX
# file to pass into groqit().
class SmallModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SmallModel, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        output = self.fc(x)
        return output


# Instantiate PyTorch model and generate inputs
input_size = 10
output_size = 5
pytorch_model = SmallModel(input_size, output_size)
onnx_model = "small_onnx_model.onnx"
input_tensor = torch.rand(input_size)
inputs = {"input": input_tensor}

# Export PyTorch Model to ONNX
torch.onnx.export(
    pytorch_model,
    input_tensor,
    onnx_model,
    opset_version=13,
    input_names=["input"],
    output_names=["output"],
)

# You can use numpy arrays as inputs to our ONNX model
def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


# Setup OnnxRuntime session for ONNX model so that you can
# present a CPU baseline for the ONNX model inference
ort_sess = ort.InferenceSession(onnx_model)
input_name = ort_sess.get_inputs()[0].name
numpy_inputs = to_numpy(input_tensor)

# Build ONNX model
groq_model = groqit(onnx_model, inputs, build_name="hello_onnx_world")

# Remove intermediate onnx file so that you don't pollute your disk
if os.path.exists(onnx_model):
    os.remove(onnx_model)

# Compute ONNX and Groq results
onnx_outputs = ort_sess.run(None, {input_name: numpy_inputs})
groq_outputs = groq_model.run(inputs)

# Print ONNX and Groq results
print(f"Groq_outputs: {groq_outputs}")
print(f"Onnx_outputs: {onnx_outputs}")
