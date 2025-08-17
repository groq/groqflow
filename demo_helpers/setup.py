from setuptools import setup, find_packages

setup(
    name="groqflow_demo_helpers",
    version="0.2.0",
    description="Helper functions to run GroqFlow demos and proof points",
    author="Groq",
    author_email="sales@groq.com",
    license="groq-license",
    packages=find_packages(
        exclude=["*.__pycache__.*"],
    ),
    include_package_data=True,
    install_requires=[
        "charset-normalizer==3.3.2",
        "transformers>=4.20.0",
        "datasets>=2.3.2",
        "prettytable>=3.3.0",
        "wget>=3.2",
        "setuptools==65.5.1",
        "torchvision==0.16.0",
        "torchaudio==2.1.0",
        "path>=16.4.0",
    ],
    classifiers=[],
    entry_points={},
)
