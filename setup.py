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
        "mlagility==3.3.1",
        "onnx==1.14.0",
        "onnxruntime==1.15.1",
        "protobuf==3.20.3",
        "scikit-learn==1.1.1",
        "torch==2.1.0",
        "typeguard==4.0.0",
    ],
    extras_require={
        "tensorflow": ["tensorflow-cpu>=2.8.1", "tf2onnx>=1.12.0"],
    },
    classifiers=[],
    python_requires=">=3.8, <3.11",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)
