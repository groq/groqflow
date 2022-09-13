"""
    This example shows how to build a small model with
    a list of assembler flags. Valid assembler flags can be found
    in the Compiler User Guide on the customer portal at
    support.groq.com

    If a list of assembler flags is provided to groqit(), then the
    default flags are not used. Any of the default flags needed
    should also be provided.

    To check the assembler flags used in a build, you can either print the
    value of the 'gmodel.state.info.assembler_command' or view the yaml file
    in the cache directory for your build.
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
user_provided_assembler_flags = ["--ifetch-from-self", "--no-metrics"]

# Build model with user-provided assembler flags
gmodel = groqit(pytorch_model, inputs, assembler_flags=user_provided_assembler_flags)

# Print the user-provided flags and the Groq Assembler command
# to verify your flags were applied.
print(f"\nUser-provided flags: {user_provided_assembler_flags}")
print(f"Groq Assembler command: {gmodel.state.info.assembler_command}")
