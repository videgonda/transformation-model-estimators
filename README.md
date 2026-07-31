# Transformation Models — Thesis Code

Semiparametric rank estimation of the additive transformation model

    Lambda_0(Y) = X beta + epsilon

comparing the estimators of Chen (2002, *Econometrica*) and Ye & Duan
(1997, *Annals of Statistics*) against engression (Shen & Meinshausen,
2024), in a Monte Carlo study and an air-quality application.

## Folder structure

```
transformation-model-estimators/
├── pyproject.toml                     installable package definition
├── src/transformation_models/         the estimator library
│   ├── chen_estimator.py              Chen (2002) rank estimator
│   ├── ye_duan_estimator.py           Ye & Duan (1997) shifted-quantile estimator
│   ├── beta_estimators.py             first-step index estimator (Han's MRC)
│   └── __init__.py
├── data/AirQualityUCI.csv             raw dataset for the application
├── simulations/
│   ├── monte_carlo/                    Monte Carlo experiment (Chapter: simulations)
│   │   ├── chen_monte_carlo.py         designs + Chen MC   (defines DESIGNS, generate_data)
│   │   ├── ye_duan_monte_carlo.py      Ye & Duan MC
│   │   ├── engression_monte_carlo.py   engression MC (needs torch + engression)
│   │   ├── combine_plots.py            merges the per-design panels
│   │   ├── combine_rmse.py             builds rmse_comparison / bias_comparison
│   │   ├── run_engression.sh           runs the tuned engression config for all designs
│   │   ├── *_design_*_results.csv      pointwise results (mean, bias, sd, rmse)
│   │   └── *.png                       generated figures
│   ├── air_quality_application/        real-data application
│   │   ├── air_quality_application.py  fits all estimators on the air-quality data
│   │   ├── plot_air_quality_results.py plots Lambda_hat, F_hat, scatter, per-estimator Lambda-vs-z
│   │   └── combine_lambda_vs_z.py      combined Lambda-vs-z figure
│   └── results/                        application outputs (CSVs + PNGs)
└── figures/                           copies of the figures used in the thesis
```

`figures/` is a convenience collection; every plot it contains is also
produced in `simulations/monte_carlo/` or `simulations/results/` by the
scripts.

## Setup

```bash
python3 -m pip install --upgrade pip        # editable install requires pip >= 21.3
python3 -m pip install -e .                 # core: numpy, scipy, matplotlib
python3 -m pip install -e ".[engression]"   # + torch, engression (engression estimator: MC + application)
```

## Reproducing the results

All scripts write their outputs next to themselves (Monte Carlo) or into
`simulations/results/` (application). Run them from the repository root.

### Monte Carlo (n = 1000, 1000 replications)

```bash
# Chen and Ye–Duan, per design (repeat with --design 1, 2, 3)
python3 simulations/monte_carlo/chen_monte_carlo.py    --design 1 --n_reps 1000
python3 simulations/monte_carlo/ye_duan_monte_carlo.py --design 1 --n_reps 1000

# engression, tuned configuration for all three designs (run_engression.sh defaults to 1000 reps)
bash simulations/monte_carlo/run_engression.sh

# comparison figures across the three estimators
python3 simulations/monte_carlo/combine_plots.py
python3 simulations/monte_carlo/combine_rmse.py
```

Design 1 = identity, 2 = logarithmic, 3 = hyperbolic sine; the evaluation
grids are chosen so that Lambda_0(y) spans [-3, 3] in every design.

### Air-quality application

```bash
python3 simulations/air_quality_application/air_quality_application.py
python3 simulations/air_quality_application/plot_air_quality_results.py
python3 simulations/air_quality_application/combine_lambda_vs_z.py
```

The application reads `data/AirQualityUCI.csv` by default; override with
`--csv <path>`.

## References

- S. Chen (2002). Rank Estimation of Transformation Models.
  *Econometrica* 70(4), 1683–1697.
- Z. Ye and N. Duan (1997). Nonparametric `n^{-1/2}`-Consistent Estimation
  for the General Transformation Models. *Annals of Statistics* 25(6), 2682–2717.
- X. Shen and N. Meinshausen (2024). Engression: extrapolation through the lens
  of distributional regression
```
