# Kaggle Playground S6E10 — Predicting Airline Satisfaction

Binary classification (satisfied vs. neutral/dissatisfied passengers) on the synthetic tabular data of
[Kaggle Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10).
Metric: **ROC AUC**.

The focus of this notebook is a **careful, leakage-checked baseline**: verify that the validation is trustworthy before tuning anything.

## Results

| Model | CV AUC (OOF) | Public LB |
|---|---|---|
| LightGBM | 0.95892 | — |
| XGBoost | 0.95866 | — |
| **Rank blend (LGBM + XGB)** | **0.95900** | **0.95841** |

Final private leaderboard: **TODO — rank / number of teams (top X %)**

## Approach

1. **Sanity and leakage checks**
   - AUC of the `id` column alone: **0.500** → row order carries no target information.
   - No duplicate rows.
   - **Adversarial validation** (train vs. test classifier): AUC **0.499** → train and test come from the same distribution, so a standard stratified K-fold CV can be trusted.
2. **Models**: gradient-boosted trees (LightGBM, XGBoost) with stratified K-fold cross-validation and out-of-fold predictions.
3. **Ensembling**: rank-average blend of the two models' predictions (AUC only depends on ranking).
4. **CV vs. leaderboard**: CV 0.95900 vs. public 0.95841 — a small, consistent gap, as expected after the adversarial check.

## Repository content

```
├── README.md
├── requirements.txt
├── .gitignore
└── s6e10_careful_baseline.ipynb   # full pipeline: checks → CV → blend → submission
```

## How to run

1. Download the competition data from the Kaggle page (it is **not** included in this repository, per competition rules) and place the CSV files in `data/`.
2. `pip install -r requirements.txt`
3. Run `s6e10_careful_baseline.ipynb` top to bottom. It writes `submission.csv`.

The notebook also runs as-is on Kaggle (data under `/kaggle/input/`).

## Possible improvements

- Add the original airline passenger satisfaction dataset to the training data (reported gain ≈ +0.002 AUC in public notebooks).
- CatBoost as a third model; Optuna tuning.
