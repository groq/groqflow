"""
    This example illustrates how to get the estimated performance of your build using the
    method `GroqModel.estimate_performance()`. You can read the details of
    `estimate_performance()` in the Performance Estimation section in docs/user_guide.md.
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

# Get performance estimates in terms of latency and throughput
estimate = gmodel.estimate_performance()
print("Your build's estimated performance is:")
print(f"{estimate.latency:.7f} {estimate.latency_units}")
print(f"{estimate.throughput:.1f} {estimate.throughput_units}")
