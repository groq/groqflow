import json
from setuptools import setup, find_packages

with open("groqflow/version.json", "r", encoding="utf8") as stream:
    json_data = stream.read()
    params = json.loads(json_data)

setup(
    name="groqflow",
    version=params["version"],
    description="GroqFlow toolchain library",
    url="https://github.com/groq/groqflow",
    author="Groq",
    author_email="sales@groq.com",
    license="Groq License Agreement",
    packages=find_packages(
        exclude=["*.__pycache__.*"],
    ),
    install_requires=[
        "onnx==1.11.0",
        "onnxmltools==1.10.0",
        "onnxruntime==1.10.0",
        "paramiko==2.11.0",
        "torch>=1.12.1",
        "protobuf==3.19.4",
        "pyyaml==6.0",
        "tensorflow-cpu>=2.9.1",
        "tf2onnx>=1.12.0",
        "typeguard>=2.3.13",
        "packaging",
    ],
    classifiers=[],
    entry_points={
        "console_scripts": [
        ]
    },
)
