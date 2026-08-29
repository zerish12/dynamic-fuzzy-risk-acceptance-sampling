## Running the analysis

After starting Jupyter Notebook, open:

`Dynamic_Fuzzy_Risk_Acceptance_Sampling_A_to_Z_Lab_FIXED.ipynb`

Then use:

**Kernel -> Restart & Run All**

The notebook performs the complete computational workflow from source-data verification through the final reproducibility checkpoints.

## Primary frozen specification

The primary analysis uses:

- Acceptable quality level: `pA = 0.01`
- Rejectable quality level: `pR = 0.03`
- History window: `K = 10`
- Primary minimum episode size: `Nmin = 50`
- Primary episode gap: `24 hours`
- Producer fuzzy tolerance: `aP = 0.20`, `bP = 0.40`
- Consumer baseline tolerance: `bC = 0.40`
- Consumer minimum tolerance: `bC,min = 0.20`
- Minimum producer satisfaction: `etaP = 0.10`
- Minimum consumer satisfaction: `etaC = 0.10`
- Sampling-cost coefficient: `kappa = 0.001`
- Satisfaction-loss coefficient: `lambda = 1`
- Producer/consumer weights: `wP = wC = 0.50`
- Primary retrospective replays: `1,000`
- Primary random seed: `20260828`

## Deterministic plan checkpoints

The validated workflow reproduces the principal deterministic sampling-plan checkpoints, including:

- Fixed classical comparator: `(n, c) = (38, 0)`
- Static fuzzy comparator: `(n, c) = (142, 2)`
- Proposed dynamic fuzzy method:
  - 584 ordinary-sampling episodes
  - 78 escalated-inspection episodes

For the proposed method, the acceptance-number distribution among ordinary-sampling episodes is:

- `c = 0`: 509 episodes
- `c = 1`: 63 episodes
- `c = 2`: 10 episodes
- `c = 3`: 2 episodes

## Retrospective replay

The primary retrospective evaluation uses 1,000 paired replays with common randomized within-episode ordering across methods.

The principal methods compared are:

1. Dynamic crisp sampling
2. Fixed classical sampling
3. Proposed dynamic fuzzy-risk sampling
4. Static fuzzy sampling

The replay analysis evaluates false-acceptance rate, false-rejection rate, average sample number, inspection fraction, and escalation behaviour.

## Robustness analysis

The reproducibility notebook includes two principal robustness blocks.

**Robustness Block A** examines combinations of:

- `pR = {0.02, 0.03, 0.05}`
- `K = {5, 10, 20}`
- `Nmin = {30, 50}`

**Robustness Block B** reconstructs production episodes using:

- 12-hour gap
- 24-hour gap
- 48-hour gap

These analyses assess whether the substantive findings depend strongly on the primary specification.

## Reproducibility note

The validated Jupyter notebook is the primary executable reproducibility workflow. A separate rendered results file is included as a human-readable record of the executed analysis.

Deterministic data-processing, production-episode construction, classification, and sampling-plan checkpoints are expected to reproduce exactly when the verified source dataset is used.

Replay-based operating characteristics are Monte Carlo estimates. Consequently, very small last-decimal differences may occur across computational environments while leaving the substantive findings and conclusions unchanged.

## Data and code availability

The raw source dataset is not redistributed in this repository. Researchers should obtain `PROD_DS_PD.csv` from the publicly available Kaggle dataset and verify it using the SHA-256 fingerprint reported above.

The reproducibility materials are maintained in this GitHub repository and permanently archived through Zenodo.

Archived release DOI:

**10.5281/zenodo.22165440**

## Citation

If using the reproducibility materials, please cite the archived Zenodo release and the associated article.

Citation metadata are also provided in `CITATION.cff`.

## Licence

The reproducibility code in this repository is released under the MIT License.

The MIT License applies to the repository code and does not grant rights to redistribute the externally hosted Kaggle dataset.
