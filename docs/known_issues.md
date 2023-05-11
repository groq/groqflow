# GroqFlow Known Issues

* Runtime errors due to mismatches in tensor sizes may occur even though GroqFlow checks the data shape. (G14148)
* Whacky terminal line wrapping when printing groqit error messages. (G13235)
* GroqFlow requires both the runtime and developer package to be installed. (G18283, G18284)
* GroqFlow BERT Quantization Proof Point fails to compile in SDK0.9.3 due to a scheduling error. (G16739)
* Yolo v6 Proof Points fails to run the evaluation after compilation in SDK0.9.2.1. (G18209)