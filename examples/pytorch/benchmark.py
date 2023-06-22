"""
    This example illustrates how to get benchmarked performance of your build on a GroqNode
    system using the method `GroqModel.benchmark()`. You can read the details of
    `benchmark()` in the Benchmark section in docs/user_guide.md.
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
gmodel = groqit(pytorch_model, inputs, groqview=True)

# Get benchmarked performance in terms of latency and throughput
performance = gmodel.benchmark()
print("Your build's estimated performance is:")
print(f"{performance.latency:.7f} {performance.latency_units}")
print(f"{performance.throughput:.1f} {performance.throughput_units}")

print("Example benchmark.py finished")
