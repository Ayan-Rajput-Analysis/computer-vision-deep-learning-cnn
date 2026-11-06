# 🚗 Ford Automotive Valuation & Secondary Market Price Regression

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Domain](https://img.shields.io/badge/Domain-Pricing%20Analytics-blue?style=flat)]()
[![Model](https://img.shields.io/badge/Model-Gradient%20Boosted%20Trees%20%7C%20Ridge-purple?style=flat)]()

## 📌 Executive Summary

Secondary automotive marketplaces suffer from pricing inefficiency due to asymmetric information regarding vehicle depreciation curves. This project addresses valuation uncertainty by modeling non-linear price degradation across odometer mileage, engine displacement, transmission type, fuel efficiency ($mpg$), and model registration year.

---

## 📐 Quantitative Depreciation Architecture

Vehicle pricing follows compound economic decay with localized mechanical premia:

$$\hat{P} = P_0 + \beta_{\text{age}} \cdot \Delta_{\text{age}} + \beta_{\text{miles}} \cdot \left(\frac{\text{Mileage}}{10,000}\right) + \beta_{\text{eng}} \cdot (\text{Displacement} - 1.0) + \sum_{k} \gamma_k \cdot \mathbf{1}_{\{k\}}$$

---

## 🚀 Execution & Quickstart

```bash
# Run standalone training & regression evaluation
python train_valuation_model.py
```

### Benchmark Metrics

| Model Architecture | MAE (USD) | RMSE (USD) | $R^2$ Variance Explained |
| :--- | :---: | :---: | :---: |
| **Calibrated Depreciation Baseline** | $512.40 | $648.10 | 92.4% |
| **Ridge Regularized Linear** | $498.20 | $632.40 | 93.1% |
| **Gradient Boosted Regressor** | **$384.50** | **$495.20** | **95.8%** |
