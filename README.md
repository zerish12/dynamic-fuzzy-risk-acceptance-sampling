# Dynamic Fuzzy-Risk Acceptance Sampling — A-to-Z Reproducibility Package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22165440.svg)](https://doi.org/10.5281/zenodo.22165440)

This repository provides the reproducibility materials for the article:

**Dynamic Fuzzy-Risk Acceptance Sampling Using Recent Production History**

The repository contains the Python and Jupyter workflow used for data preprocessing, production-episode construction, sampling-plan optimisation, comparator analysis, retrospective replay, decision-loss analysis, robustness analysis, and reproduction of the principal manuscript results.

## Dataset verification

The source CSV used for the article was independently verified against the reproducibility workflow.

- rows: 80,015
- columns: 28
- encoding: UTF-16
- SHA-256: `613d38f03dd82aa2db6ddec7826719861461104b1dd4188369442dbe8477da3f`

The verified deterministic data chain is:

`80,015 -> 80,000 -> 74,685 -> 1,227 episodes -> 662 primary episodes`

The primary episode classes are:

`404 acceptable / 160 indifference / 98 rejectable`

## Dataset source

The empirical analysis uses the publicly available **Product Data CSV** dataset hosted on Kaggle.

- uploader: Tadewos Bellete
- Kaggle identifier: `tadewosbellete/product-data-csv`
- source file required by the analysis: `PROD_DS_PD.csv`

The raw CSV is intentionally not redistributed with this repository because its redistribution licence has not been independently verified.

Place the source file at:

`data/raw/PROD_DS_PD.csv`

If the file is absent, the notebook will also attempt to retrieve the dataset using `kagglehub`.

## Installation and execution

Create and activate a Python virtual environment and install the required dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
