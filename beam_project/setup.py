from setuptools import setup, find_packages

setup(
    name="beam_simulation",
    version="0.1.0",
    author="Manon Bart",
    author_email="manon.bart.p@gmail.com",
    description="A Python package for simulating LG, HG, and IG beams with turbulence and propagation.",
    url="https://github.com/manonpbart/Machine-Learning-for-Optical-Communication/",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
