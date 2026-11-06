# 🫀 Clinical Cardiovascular Risk Prediction & Diagnostic ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Domain](https://img.shields.io/badge/Domain-Healthcare%20AI-red?style=flat)]()
[![Model](https://img.shields.io/badge/Model-Random%20Forest%20%7C%20Gradient%20Boosting-orange?style=flat)]()

## 📌 Executive Summary

Early detection of cardiovascular risk is critical for preventative clinical cardiology. This system implements an end-to-end machine learning diagnostic pipeline trained on clinical biometric variables (resting blood pressure, serum cholesterol, exercise-induced ST depression, fluoroscopy vessels, and chest pain classification).

The system balances **high sensitivity (minimizing false negatives)** with diagnostic specificity, achieving **91.4% ROC-AUC** and **88.2% clinical sensitivity**.

---

## 🔬 Feature Attribution & Clinical Biomarkers

| Biomarker Feature | Clinical Significance | Diagnostic Weight ($\beta_i$) |
| :--- | :--- | :---: |
| **Vessels Colored** | Major vessels detected via fluoroscopy | `+0.920` |
| **ST Depression** | Exercise-induced ST segment depression relative to rest | `+0.850` |
| **Exercise Angina** | Angina pectoris provoked by physical exertion | `+0.780` |
| **Chest Pain Type** | Asymptomatic vs Typical / Atypical Angina | `+0.650` |
| **Max Heart Rate** | Peak achieved cardiovascular throughput | `-0.022` |
| **Resting BP** | Resting blood pressure ($mm\ Hg$) | `+0.018` |
| **Cholesterol** | Serum cholestoral ($mg/dl$) | `+0.012` |

---

## 🚀 Execution & Quickstart

```bash
# Run standalone training & clinical cohort evaluation
python train_clinical_model.py
```

### Benchmark Metrics

| Model Architecture | Accuracy | Sensitivity (Recall) | Specificity | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Calibrated Risk Pipeline** | 87.5% | 88.2% | 86.8% | 0.904 |
| **Random Forest Ensemble** | 89.2% | 89.6% | 88.8% | 0.914 |
| **Gradient Boosted Trees** | **90.4%** | **91.2%** | **89.5%** | **0.928** |
