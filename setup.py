from setuptools import find_packages, setup

setup(
    name="tomeov",
    version="0.1.0",
    author="Alexander Kozlov",
    url="https://github.com/AlexKoff88/tomesd",
    description="Token Merging for OpenVINO",
    install_requires=["torch~=1.13", "onnx", "onnxruntime", "accelerate", "diffusers", "openvino", "optimum", "optimum-intel"],
    packages=find_packages(exclude=("examples", "build")),
    license = 'Apache 2.0',
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)