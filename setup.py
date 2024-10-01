from setuptools import setup, find_packages

setup(
    name="ocr_model_benchmark",
    version="0.1.0",
    packages=find_packages(
        include=["ocr_benchmark", "ocr_benchmark.*"]
    ),
    description="Benchmark in python of the different\
        OCR methods",
    author="Hippolyte Guigon",
    author_email="Hippolyte.guigon@hec.edu",
)