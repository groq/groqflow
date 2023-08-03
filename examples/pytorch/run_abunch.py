"""
    Hello World, again!

    This example uses the same small model as the hello_world example,
    but this time we are going to run a bunch of inferences with the
    GroqModel.run_abunch() method.
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

# Compile model
groq_model = groqit(pytorch_model, inputs)

# Create a bunch of inputs
num_inputs = 10
abunch_o_inputs = [{"x": torch.rand(input_size)} for _ in range(num_inputs)]

print(f"Calculating the results of the {num_inputs} inputs!")

# Run groq_model computations on abunch_o_inputs
abunch_o_outputs = groq_model.run_abunch(input_collection=abunch_o_inputs)

# Print abunch of outputs
for count, output in enumerate(abunch_o_outputs):
    print(f"output {count}: {list(output.numpy())}")

print("Example run_abunch.py finished")
