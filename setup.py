from setuptools import find_packages, setup

setup(
    name="tomesd",
    version="0.1",
    author="Daniel Bolya",
    url="https://github.com/dbolya/tomesd",
    description="Token Merging for Stable Diffusion",
    install_requires=["torch"],
    packages=find_packages(exclude=("examples", "build")),
)