"""
    This example shows how to specify the number of GroqChip processors
    used in your build.

    You will need to be able to put at least one layer on each chip. So, the
    small model here will have two layers.

    To check the number of chips used in a build, you can either print the
    value of the 'gmodel.state.num_chips_used' or view the yaml file
    in the cache directory for your build.

    You can read more about the `num_chips` argument and multi-chip builds
    in the Multi-Chip section in the docs/user_guide.md.
"""

import torch
from groqflow import groqit

torch.manual_seed(0)

# Define model class
class TwoLayerModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(TwoLayerModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, output_size)
        self.fc2 = torch.nn.Linear(output_size, output_size)

    def forward(self, x):
        output = self.fc1(x)
        output = self.fc2(output)
        return output


# Create model and inputs
input_size = 10
output_size = 5
pytorch_model = TwoLayerModel(input_size, output_size)
inputs = {"x": torch.rand(input_size)}

# Build model for 2 chips
gmodel = groqit(pytorch_model, inputs, num_chips=2)

print(
    "\nThe number of GroqChip processors required to run the build is "
    f"{gmodel.state.num_chips_used}."
)

print("Example num_chips.py finished")
