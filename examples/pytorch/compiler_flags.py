"""
    This example shows how to build a small model with
    a list of compiler flags. Valid compiler flags can be found
    in the Compiler User Guide on the customer portal at
    support.groq.com

    If a list of compiler flags is provided to groqit(), then the
    default flags are not used. Any of the default flags needed
    should also be provided.

    To check the compiler flags used in a build, you can either print the
    value of the 'gmodel.state.info.compiler_command' or view the yaml file
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
user_provided_compiler_flags = ["--no-print-stats", "--disableAddressCompaction"]

# Build model with user provided compiler flags
gmodel = groqit(pytorch_model, inputs, compiler_flags=user_provided_compiler_flags)

# Print the user-provided flags and the Groq Compiler command
# to verify your flags were applied.
print(f"\nUser-provided flags: {user_provided_compiler_flags}")
print(f"Groq Assembler command: {gmodel.state.info.compiler_command}")

print("Example compiler_flags.py finished")
