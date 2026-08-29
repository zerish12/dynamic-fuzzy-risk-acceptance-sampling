# Dynamic Fuzzy-Risk Acceptance Sampling — A-to-Z Reproducibility Package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22165440.svg)](https://doi.org/10.5281/zenodo.22165440)

This repository provides the reproducibility materials for the article:

**Dynamic Fuzzy-Risk Acceptance Sampling Using Recent Production History**

The repository contains the validated Python/Jupyter workflow used for data preprocessing, production-episode construction, sampling-plan optimisation, comparator analysis, retrospective replay, decision-loss analysis, robustness analysis, and reproduction of the principal manuscript results.

## Repository contents

The principal reproducibility materials are:

- `Dynamic_Fuzzy_Risk_Acceptance_Sampling_A_to_Z_Lab_FIXED.ipynb`  
  Primary executable Jupyter notebook containing the complete A-to-Z computational workflow.

- `python_codes_results_Dynamic_Fuzzy_Risk...`  
  Human-readable record of the Python code and executed analysis results.

- `DATASET_VERIFICATION.txt`  
  Source-data identity and deterministic verification information.

- `requirements.txt`  
  Python dependencies required to run the notebook.

- `CITATION.cff`  
  Citation metadata for the reproducibility package.

- `LICENSE`  
  MIT licence covering the code in this repository.

The validated Jupyter notebook is the primary executable reproducibility artifact.

## Dataset verification

The source CSV used for the article was independently verified against the reproducibility workflow.

- Rows: 80,015
- Columns: 28
- Encoding: UTF-16
- SHA-256: `613d38f03dd82aa2db6ddec7826719861461104b1dd4188369442dbe8477da3f`

The verified deterministic data chain is:

`80,015 -> 80,000 -> 74,685 -> 1,227 episodes -> 662 primary episodes`

The primary episode classes are:

`404 acceptable / 160 indifference / 98 rejectable`

Additional deterministic checkpoints include:

- 5,315 exact duplicate raw records removed
- 72,885 unique serial numbers after cleaning
- 866 original lot identifiers
- 933 defective records
- 1,227 production episodes under the primary 24-hour gap rule
- 662 primary eligible episodes with `N >= 50`
- 584 ordinary-sampling episodes under the proposed method
- 78 escalated-inspection episodes

## Dataset source

The empirical analysis uses the publicly available **Product Data CSV** dataset hosted on Kaggle.

- Uploader: Tadewos Bellete
- Kaggle identifier: `tadewosbellete/product-data-csv`
- Required source file: `PROD_DS_PD.csv`

The raw CSV is intentionally not redistributed with this repository because its redistribution licence has not been independently verified.

For local reproduction, the source file may be placed at:

`data/raw/PROD_DS_PD.csv`

The validated notebook also searches common local locations for the source file and verifies the file against the expected SHA-256 fingerprint before beginning the analysis.

If the verified source file is not available locally, the notebook attempts retrieval using `kagglehub`.

## Installation

A clean Python environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
