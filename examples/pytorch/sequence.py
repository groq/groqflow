""" This example uses GroqFlow features recommended for power users only.

    By default, GroqFlow completes the following steps:
     > Convert to ONNX
     > Optimize ONNX file
     > Check op support
     > Convert to FP16
     > Compile Model
     > Assemble Model

    This example illustrates how to alter the default sequence of steps. In this
    example, the conversion to FP16 is skipped.
"""

import torch
from groqflow import groqit
import onnxflow.justbuildit.export as of_export
import onnxflow.justbuildit.stage as stage
import groqflow.justgroqit.compile as compile
import groqflow.justgroqit.export as gf_export


torch.manual_seed(0)

# Define model class
class SmallModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SmallModel, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        output = self.fc(x)
        return output


# Instantiate model and generate inputs
input_size = 10
output_size = 5

pytorch_model = SmallModel(input_size, output_size)
inputs = {"x": torch.rand(input_size, dtype=torch.float32)}

onnx_sequence = stage.Sequence(
    "onnx_sequence",
    "Building ONNX Model without fp16 conversion",
    [
        of_export.ExportPytorchModel(),
        of_export.OptimizeOnnxModel(),
        gf_export.CheckOnnxCompatibility(),
        # of_export.ConvertOnnxToFp16(),  #<-- This is the step we want to skip
        compile.CompileOnnx(),
        compile.Assemble(),
    ],
    enable_model_validation=True,
)

# Build model
groq_model = groqit(pytorch_model, inputs, sequence=onnx_sequence)

# Compute Pytorch and Groq results
pytorch_outputs = pytorch_model(**inputs)
groq_outputs = groq_model(**inputs)

# Print Pytorch and Groq results
print(f"Pytorch_outputs: {pytorch_outputs}")
print(f"Groq_outputs: {groq_outputs}")
