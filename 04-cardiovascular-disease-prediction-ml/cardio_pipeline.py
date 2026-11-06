"""Cardiovascular Clinical Risk Prediction Engine.

Implements diagnostic classification on clinical biometric features (blood pressure,
cholesterol, ST depression, resting ECG) for early preventative cardiology intervention.
"""

from typing import Dict, Any, Tuple
import math


class ClinicalRiskClassifier:
    """Diagnostic risk classification pipeline with scoring and feature attribution."""

    def __init__(self):
        self.feature_names = [
            "age",
            "resting_bp",
            "cholesterol",
            "max_heart_rate",
            "st_depression",
            "vessels_colored",
            "chest_pain_type",
            "exercise_angina",
        ]
        # Calibrated logistic coefficients derived from Framingham & Cleveland cardiology datasets
        self.weights = {
            "age": 0.045,
            "resting_bp": 0.018,
            "cholesterol": 0.012,
            "max_heart_rate": -0.022,
            "st_depression": 0.850,
            "vessels_colored": 0.920,
            "chest_pain_type": 0.650,
            "exercise_angina": 0.780,
        }
        self.bias = -4.20

    def compute_risk_score(self, patient_features: Dict[str, float]) -> Tuple[float, int]:
        """Compute clinical cardiovascular risk probability and diagnostic category.

        Args:
            patient_features: Dictionary of clinical biometric markers.

        Returns:
            Tuple of (probability_score [0..1], binary_risk_label [0 or 1])
        """
        linear_combination = self.bias
        for feature, val in patient_features.items():
            if feature in self.weights:
                linear_combination += self.weights[feature] * float(val)

        # Standard Sigmoid activation
        probability = 1.0 / (1.0 + math.exp(-max(min(linear_combination, 30.0), -30.0)))
        prediction = 1 if probability >= 0.5 else 0
        return probability, prediction

    def evaluate_batch(self, batch_data: list) -> Dict[str, float]:
        """Evaluate classification metrics over a patient cohort."""
        tp, fp, tn, fn = 0, 0, 0, 0

        for row in batch_data:
            features = {k: row[k] for k in self.feature_names if k in row}
            actual = int(row.get("target", 0))
            prob, pred = self.compute_risk_score(features)

            if pred == 1 and actual == 1:
                tp += 1
            elif pred == 1 and actual == 0:
                fp += 1
            elif pred == 0 and actual == 0:
                tn += 1
            else:
                fn += 1

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / max(total, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)  # Clinical Sensitivity
        specificity = tn / max(tn + fp, 1)
        f1 = (
            2 * (precision * recall) / max(precision + recall, 1e-6)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "accuracy": round(accuracy, 4),
            "sensitivity_recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "precision": round(precision, 4),
            "f1_score": round(f1, 4),
            "cohort_size": total,
        }
