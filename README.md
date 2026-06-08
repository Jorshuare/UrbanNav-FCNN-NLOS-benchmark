# UrbanNav FCNN — GNSS LOS/NLOS & Multipath Benchmark (Reproduction)

A clean-room, reproducible re-implementation of the **FCNN benchmark** from:

> P. Xu, G. Zhang, B. Yang, and L.-T. Hsu, **"Machine Learning in GNSS
> Multipath/NLOS Mitigation: Review and Benchmark,"** *IEEE Aerospace and
> Electronic Systems Magazine*, 39(9):26–43, Sep. 2024.
> DOI: [10.1109/MAES.2024.3395182](https://doi.org/10.1109/MAES.2024.3395182).
> The Hong Kong Polytechnic University.

GNSS positioning degrades in urban canyons because buildings block (NLOS) or
reflect (multipath) satellite signals. This repo reproduces the paper's three
benchmark tasks on the UrbanNav Hong Kong dataset using a Fully Connected Neural
Network (FCNN) and an ordinary-least-squares (OLS) positioning baseline.

## Tasks & Kaggle competitions

| # | Task | Model | Metric | Kaggle competition |
|---|------|-------|--------|--------------------|
| 1 | LOS/NLOS classification | FCNN (M2: 64-256-64) | Accuracy | [gnss-classification](https://www.kaggle.com/c/gnss-classification) |
| 2 | Pseudorange error prediction | FCNN (M4: 128-128-64) | RMSE (m) | [PrError-Prediction](https://www.kaggle.com/c/PrError-Prediction) |
| 3 | Positioning error | OLS least squares | RMSE E/N (m) | [gnss-urban-positioning](https://www.kaggle.com/c/gnss-urban-positioning) |

## Results

Validation numbers are reproduced **locally**; the held-out test scores come from
the **Kaggle leaderboard** (the test labels are not public).

| Task | Ours (validation) | Ours (Kaggle test) | Paper (val) | Paper (test) |
|------|------------------:|-------------------:|------------:|-------------:|
| Classification (M2) | 0.828 (0.8265 ± 0.0027, 10 seeds) | **0.828** | 0.771 | 0.853 |
| Pseudorange RMSE (M4) | 16.48 m (16.32 ± 0.09) | — (competition gated) | 15.14 | 13.01 |
| Positioning OLS RMSE | 15.99 m | — (competition gated) | 17.76 | 21.49 |

- **Classification reproduced end-to-end** to **97% of the published benchmark**
  (Kaggle test 0.828 vs 0.853). The single-split model scores 0.812; retraining
  on all labeled data + a 10-seed probability ensemble lifts it to 0.828.
- **Regression & positioning** are reproduced on the validation split; their test
  benchmarks are only obtainable via Kaggle and those competitions were not
  accessible at reproduction time.

### Honest reproduction notes
- **Uncorrected baseline RMSE** is 25.96 m (our seed-42 split) vs the paper's
  23.87 m. A 200-seed sweep shows 23.87 m sits near the *floor* of the split
  distribution, implying the authors applied outlier/preprocessing we couldn't
  observe. We keep seed 42 and report transparently rather than seed-pick.
- **The five architecture variants are statistically indistinguishable** on this
  data (val-acc spread 0.005, val-RMSE spread 0.23 m, within seed noise), so the
  paper's specific "M2-best / M4-best" selections fall within run-to-run variation.
- We use the authors' **precomputed features** (`CNR, Elevation, Pr_Residual`);
  the raw RINEX CSVs lack ephemeris, so these cannot be derived from them alone.

## Data (not included)

The UrbanNav/Kaggle dataset is **licensed and not redistributed here**. Download
the files from the Kaggle competitions above and place them as:

```
data/raw/
├── GNSS_train.xlsx        # processed features + labels (TST+WP+MK)
├── GNSS_test.xlsx         # processed features only (KLB, no labels)
├── GNSS_raw_train.csv     # RINEX-level source
├── GNSS_raw_test.csv
└── sample_submission.csv
```

## Quick start

```bash
make setup     # uv venv (Python 3.12) + pinned deps from requirements.lock
make data      # build model-ready features -> data/processed/
make train     # train canonical M2 (clf) and M4 (reg)
make eval      # reproduce Table 8 + Table 9 (validation) + submission.csv
make figures   # training curves + Fig. 13 (C/N0 vs elevation, LOS/NLOS)
make test      # pytest (data invariants, scaler, model shapes, metrics, training)
```

## Layout

```
configs/   Hydra configs (data, features, model M1–M5, train, eval)
src/gnss_fcnn/   data · models · training · evaluation · viz · utils
scripts/   01_prepare_data · 02/03_train · 04_evaluate · 05_make_figures · 06_full_retrain_submit
tests/     pytest suite (31 tests)
reports/   reproduced tables + example submission
```

## Reproducibility

Pinned `requirements.lock`, `seed_everything` (Python/NumPy/torch) with
deterministic algorithms, config-driven hyperparameters (no magic numbers), and
a test suite pinning the data invariants (incl. the LOS/NLOS label polarity).

## License & attribution

Code is released under the [MIT License](LICENSE). The original paper, dataset,
and benchmark are the work of Xu, Zhang, Yang, and Hsu (The Hong Kong Polytechnic
University) and IEEE; this repository is an **independent reproduction for
research purposes** and does not redistribute the paper or the dataset.
