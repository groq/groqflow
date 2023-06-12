"""
    Hello ** PyTorch ** World!

    This example uses a small model to carry out a single vector matrix
    multiplication to demonstrate building and running a PyTorch model
    with GroqFlow.

    This example will help identify what you should expect from each groqit()
    PyTorch build. You can find the build results in the cache directory at
    ~/.cache/groqflow/hello_pytorch_world/ (unless otherwise specified).
"""

import torch
from groqflow import groqit

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
inputs = {"x": torch.rand(input_size)}

# Build model
groq_model = groqit(pytorch_model, inputs, build_name="hello_pytorch_world")

# Compute Pytorch and Groq results
pytorch_outputs = pytorch_model(**inputs)
groq_outputs = groq_model(**inputs)

# Print Pytorch and Groq results
print(f"Pytorch_outputs: {pytorch_outputs}")
print(f"Groq_outputs: {groq_outputs}")

print("Example hello_world.py finished")
