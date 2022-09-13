"""
    This example shows how to build a small model and collect the data necessary
    to visualize and profile a model using GroqView. When you run the
    `GroqModel.groqview()` method, the visualizer is opened in a web browser.
    See the GroqView User Guide at support.groq.com to read all about it.
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

# Open GroqView
gmodel.groqview()
