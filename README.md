# Dynamic Fuzzy-Risk Acceptance Sampling — A-to-Z Reproducibility Package

This package reproduces the article:

**Dynamic Fuzzy-Risk Acceptance Sampling Using Recent Production History**

## Dataset verification performed for this version

The user-supplied original CSV was inspected directly.

- rows: 80,015
- columns: 28
- encoding: UTF-16
- SHA-256: `613d38f03dd82aa2db6ddec7826719861461104b1dd4188369442dbe8477da3f`

The verified deterministic data chain is:

`80,015 -> 80,000 -> 74,685 -> 1,227 episodes -> 662 primary episodes`

and the primary classes are:

`404 acceptable / 160 indifference / 98 rejectable`.

## Dataset source

Kaggle:

- **Product Data CSV**
- uploader: Tadewos Bellete
- identifier: `tadewosbellete/product-data-csv`
- required source file: `PROD_DS_PD.csv`

The raw CSV is intentionally not bundled in the shareable ZIP because its redistribution licence has not been independently verified.

Place the file at:

`data/raw/PROD_DS_PD.csv`

The notebook will also try `kagglehub` if the file is absent.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Open:

`Dynamic_Fuzzy_Risk_Acceptance_Sampling_A_to_Z_Lab.ipynb`

and use **Restart & Run All**.

## Primary frozen specification

- `pA = 0.01`
- `pR = 0.03`
- `K = 10`
- `Nmin = 50`
- gap rule = 24 hours
- producer fuzzy interval = `[0.20, 0.40]`
- consumer upper tolerance moves from `0.40` to `0.20`
- `aC_t = 0.5*bC_t`
- `etaP = etaC = 0.10`
- inspection cost coefficient = `0.001`
- fuzzy penalty weight = `1.0`
- `wP = wC = 0.50`
- primary replays = 1,000
- primary seed = `20260828`

## Deterministic plan checkpoints

- fixed classical plan: `(38,0)`
- strict `alpha<=0.05, beta<=0.10`: `(390,7)`, implementable in 6/662 episodes
- proposed ordinary-sampling episodes: 584
- proposed escalations: 78
- dynamic crisp escalations: 30
- static fuzzy escalations: 651

## Manuscript replay targets

| Method | FA | FR | ASN | Inspection | Escalated |
|---|---:|---:|---:|---:|---:|
| Dynamic crisp | 0.1196 | 0.0057 | 35.9622 | 0.3740 | 30 |
| Fixed classical | 0.0890 | 0.0066 | 38.0000 | 0.3952 | 0 |
| Proposed dynamic fuzzy | 0.0209 | 0.0004 | 54.9502 | 0.5715 | 78 |
| Static fuzzy | 0.0004 | 0.0006 | 65.3248 | 0.6794 | 651 |

Tiny last-decimal Monte Carlo differences may occur across software/implementation environments; deterministic plan-selection checkpoints should match exactly.
