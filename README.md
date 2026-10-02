# Data Preprocessing and Data Leakage in Multivariate Time Series Forecasting

Code for the paper **"Data Preprocessing and Data Leakage in Multivariate Time Series Forecasting"**, a benchmark of machine learning, deep learning and statistical forecasting models.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/millimusa/Data-Preprocessing-and-Data-Leakage/blob/main/PM25_preprocessing_study.ipynb) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23096833.svg)](https://doi.org/10.5281/zenodo.23096833)

## Overview

The study measures how five preprocessing stages affect the accuracy of ten forecasting models on hourly air quality data (Beijing PM2.5 data set, UCI Machine Learning Repository).

| Stage | Methods compared (reference in bold) |
|---|---|
| Missing value imputation | mean, **linear interpolation**, KNN |
| Anomaly handling | **none**, z-score (3 SD), IQR (1.5 IQR), Isolation Forest |
| Categorical encoding | label, **one-hot**, target |
| Normalization | none, **min-max**, z-score, robust |
| Smoothing | **none**, moving average, exponential smoothing, Savitzky-Golay (applied to the series or to the inputs only) |

**Models:** linear regression, decision tree, random forest, XGBoost and k-nearest neighbors (machine learning); 1D-CNN, LSTM and GRU (deep learning); ARIMAX and Prophet (statistical); a persistence forecast as baseline.

One stage at a time is replaced around the reference pipeline, which gives 17 configurations and 167 model runs. The years 2010 to 2013 are used for training and 2014 for testing. Every preprocessing method learns its parameters from the training data only and, in the test year, uses only values observed before the forecast hour. Forecasts one hour ahead are scored against the original measurements with MAE, RMSE and MAPE, and each configuration is compared with the reference pipeline by the Diebold-Mariano test with Holm correction.

## Main findings

- The stages that modify the target series had the largest influence. Smoothing the series raised the MAE of every model by 5% to 93%, and statistical outlier removal by up to 35%.
- Scored against the preprocessed series instead of the original measurements, the same forecasts appeared better, with median MAE reductions of 4% to 55%. This gap between apparent and real accuracy is a form of data leakage.
- Imputation mattered most for the deep learning models, where linear interpolation outperformed mean and KNN imputation by up to 8%.
- Normalization was decisive for the recurrent networks and k-nearest neighbors but hardly affected the linear, tree and ARIMAX models, and encoding had little effect.

## How to run

**Google Colab (recommended).** Open the notebook with the badge above, select a T4 GPU (Runtime > Change runtime type) and run the single code cell. The data set is downloaded automatically from the UCI Machine Learning Repository, and missing packages are installed by the script. The full run takes about 80 to 90 minutes. All results are written to `pm25_results/` and downloaded as `pm25_results.zip` at the end.

**Local run.**

```bash
pip install -r requirements.txt
PM25_LOCAL=1 python pm25_preprocessing_study.py
```

`PM25_LOCAL=1` saves the figures to files without opening plot windows. Without internet access, place `PRSA_data_2010.1.1-2014.12.31.csv` in the working directory or set `PM25_CSV` to its path. `PM25_FAST=1` runs a short smoke test (7 configurations, one seed, two epochs) whose numbers are not comparable to the paper.

## Outputs

| File | Content |
|---|---|
| `results_all.csv` | MAE, RMSE and MAPE of all runs, apparent scores, Diebold-Mariano statistics and Holm-adjusted p values |
| `table_*.csv` | Result tables of the reference pipeline and of each stage |
| `dm_tests.csv` | Diebold-Mariano tests against the reference pipeline |
| `preprocessing_info.csv` | Missing and flagged hours and fitted anomaly thresholds per configuration |
| `sensitivity_missing.csv`, `imputation_error.csv` | Sensitivity experiment with 10%, 20% and 30% additional missing data |
| `arima_order_selection.csv` | AIC and BIC of the candidate ARIMAX orders |
| `test_forecasts.npz` | Hourly test forecasts of every run and the observed test series |
| `run_info.json` | Software versions, run time, selected ARIMAX order and stationarity test |
| `fig01_*.png` to `fig10_*.png` | Figures |

Figure files and the corresponding figures in the paper:

| File | Paper |
|---|---|
| `fig02_data.png` | Fig. 1 |
| `fig01_workflow.png` | Fig. 2 |
| `fig04_smoothing_example.png` | Fig. 3 |
| `fig07_forecast_week.png` | Fig. 4 |
| `fig05_mae_change_heatmap.png` | Fig. 5 |
| `fig09_missing_sensitivity.png` | Fig. 6 |
| `fig03_anomaly_examples.png` | Fig. 7 |
| `fig08_apparent_vs_real.png` | Fig. 8 |
| `fig06_stage_sensitivity.png` | Fig. 9 |

`fig10_reference_mae.png` is an additional overview that does not appear in the paper.

## Software

The results in the paper were obtained on Google Colab with an NVIDIA T4 GPU, Python 3.13, scikit-learn 1.6.1, XGBoost 3.4.1, TensorFlow 2.20.0, statsmodels 0.15.0 and Prophet 1.4.0. The versions are listed in `requirements.txt`.

## Data

Beijing PM2.5 data set of the UCI Machine Learning Repository, licensed under CC BY 4.0:

- Chen, S. (2015). *Beijing PM2.5* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5JS49
- Liang, X., Zou, T., Guo, B., Li, S., Zhang, H., Zhang, S., Huang, H., & Chen, S. X. (2015). Assessing Beijing's PM2.5 pollution: Severity, weather impact, APEC and winter heating. *Proceedings of the Royal Society A*, 471(2182), 20150257. https://doi.org/10.1098/rspa.2015.0257
