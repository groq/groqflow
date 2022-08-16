# GroqFlow Known Issues

## Release 2.1.1.

* If an input model is a PyTorch model with multiple inputs, GroqFlow generates an ONNX model assuming the inputs were passed as positional arguments. This can result in an incorrect graph or an error because it expected an input not provided. (G13100)
* Runtime errors due to mismatches in tensor sizes may occur even though GroqFlow checks the data shape. (G14148)
* Whacky terminal line wrapping when printing groqit error messages. (G13235)