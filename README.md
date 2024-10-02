# OCR_model_benchmark

The goal of this repository is to benchmark the most recent and popular OCR algorithms.

These models will be tested on the FUNSD Dataset (https://guillaumejaume.github.io/FUNSD/) and benchmarked on the obtained metrics

## Build Status

For the moment, the only implemented model is LayoutLM v2 (https://arxiv.org/pdf/1912.13318)

The next step of the project is to implement new models including:

* Donut (https://arxiv.org/pdf/2111.15664)
* SVTR (https://arxiv.org/pdf/2205.00159)

Throughout the project, if you see any improvements that could be made in the code, do not hesitate to reach out at
Hippolyte.guigon@hec.edu. I will b delighted to get some insights !

## Code style

The all project was coded under PEP-8 (https://peps.python.org/pep-0008/) and flake8 (https://pypi.org/project/flake8/) compliancy. Such compliance is verified during commits with pre-commits file ```.pre-commit-config.yaml```

## Installation

* This project uses a specific conda environment, to get it, run the following command: ```conda env create -f ocr_benchmark.yml```

* To install all necessary libraries, run the following code: ```pip install -r requirements.txt```

* You need an additionnal library. For this run ```./install_detectron2.sh```

* This project has its own package that is used. To get it, run the following command: ```python install setup.py```

## Screenshot

![alt text](https://github.com/HippolyteGuigon/OCR_model_benchmark/blob/main/ressources/FUNSDpng.png)

Result of LayoutLM on a FUNSD document

## How to use ?
