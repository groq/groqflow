from setuptools import setup, find_packages

with open("groqflow/version.py", encoding="utf-8") as fp:
    version = fp.read().split('"')[1]

setup(
    name="groqflow",
    version=version,
    description="GroqFlow toolchain library",
    url="https://github.com/groq/groqflow",
    author="Groq",
    author_email="sales@groq.com",
    license="MIT",
    packages=find_packages(
        exclude=["*.__pycache__.*"],
    ),
    install_requires=[
        "onnx==1.11.0",
        "onnxmltools==1.10.0",
        "hummingbird-ml==0.4.4",
        "scikit-learn==1.1.1",
        "xgboost==1.6.1",
        "onnxruntime==1.10.0",
        "paramiko==2.11.0",
        "torch>=1.12.1",
        "protobuf==3.17.3",
        "pyyaml==5.4",
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
    python_requires="==3.8.*",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)
