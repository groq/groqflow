"""
This example shows how to specify the data samples to be used to
perform post training quantization on the equivalent ONNX model
before compiling and assembling the model into a GroqModel.

You can read more about the `quantization_samples` argument
in the corresponding section in the docs/user_guide.md.
"""

import torch
import numpy as np
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


if __name__ == "__main__":

    # Create model and inputs
    input_size, output_size = 10, 5
    pytorch_model = TwoLayerModel(input_size, output_size)
    torch_tensor = torch.rand(input_size)
    inputs = {"x": torch_tensor}

    # Prepare quantization data
    # Datatype should be the same type for the model inputs, the model's expected inputs
    # and the quantization samples
    sample_size = 100
    quantization_data = [
        (np.array([np.random.rand(input_size)], dtype=np.float32))
        for _ in range(sample_size)
    ]

    # Convert pytorch model into ONNX, quantize the ONNX model and
    # convert quantized ONNX to GroqModel
    gmodel = groqit(
        pytorch_model,
        inputs,
        rebuild="always",
        quantization_samples=quantization_data,
    )

    # Inference both PyTorch model and Quantized GroqModel
    simple_pytorch_dataset = [
        inputs,
        inputs,
    ]
    groq_outputs = gmodel.run_abunch(simple_pytorch_dataset)
    with torch.no_grad():
        torch_outputs = [pytorch_model(**example) for example in simple_pytorch_dataset]

    # See if inference results match
    value_pass = all(
        [
            np.allclose(torch_outputs[i], groq_outputs[i], rtol=0.01, atol=0.001)
            for i in range(len(simple_pytorch_dataset))
        ]
    )
    match_str = "" if value_pass else "not "
    print(
        "Results of PyTorch model and quantized GroqModel do {}match.".format(match_str)
    )

    print("Example quantization.py finished")
