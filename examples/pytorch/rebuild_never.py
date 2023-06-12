"""
    This example is built to demonstrate groqit()'s rebuild = "never" setting.

    When rebuild is set to "never" groqit() will look within the cache
    for a build with a matching build_name and load it, if it exists.
    You will see a warning printed to stout if the model has changed, but the
    existing build will be loaded regardless of functionality or correctness.

    Try the following experiment.
    1. Run this script to build and save the model in cache.
    2. Run the script again, and observe the warning printed when the
       cached model is loaded even though there is a detected change.

    Note: To make sure the model changes, the random seed is not set
          for this example.
"""

import torch
from groqflow import groqit

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

# Build or load the model with rebuild="never" applied
groq_model = groqit(pytorch_model, inputs, rebuild="never")

print("Example rebuild_never.py finished")
