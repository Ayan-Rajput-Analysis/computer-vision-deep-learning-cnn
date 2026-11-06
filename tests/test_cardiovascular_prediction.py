"""Unit tests for Cardiovascular Clinical Risk Intelligence pipeline."""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "04-cardiovascular-disease-prediction-ml"))

from cardio_pipeline import ClinicalRiskClassifier
from train_clinical_model import generate_synthetic_clinical_cohort


class TestCardiovascularRiskPipeline(unittest.TestCase):
    """Test suite for clinical risk prediction logic."""

    def setUp(self):
        self.classifier = ClinicalRiskClassifier()

    def test_single_patient_risk_scoring(self):
        """Verify patient risk probability is strictly bounded in [0, 1]."""
        patient_high_risk = {
            "age": 68.0,
            "resting_bp": 175.0,
            "cholesterol": 310.0,
            "max_heart_rate": 110.0,
            "st_depression": 3.4,
            "vessels_colored": 3.0,
            "chest_pain_type": 3.0,
            "exercise_angina": 1.0,
        }
        prob, pred = self.classifier.compute_risk_score(patient_high_risk)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        self.assertIn(pred, [0, 1])
        self.assertGreater(prob, 0.70)
        self.assertEqual(pred, 1)

    def test_low_risk_patient(self):
        """Verify young asymptomatic profile produces low risk score."""
        patient_low_risk = {
            "age": 28.0,
            "resting_bp": 115.0,
            "cholesterol": 160.0,
            "max_heart_rate": 185.0,
            "st_depression": 0.0,
            "vessels_colored": 0.0,
            "chest_pain_type": 0.0,
            "exercise_angina": 0.0,
        }
        prob, pred = self.classifier.compute_risk_score(patient_low_risk)
        self.assertLess(prob, 0.40)
        self.assertEqual(pred, 0)

    def test_cohort_evaluation_metrics(self):
        """Verify batch cohort evaluation calculates valid sensitivity and specificity."""
        cohort = generate_synthetic_clinical_cohort(n_patients=100)
        self.assertEqual(len(cohort), 100)

        metrics = self.classifier.evaluate_batch(cohort)
        self.assertIn("accuracy", metrics)
        self.assertIn("sensitivity_recall", metrics)
        self.assertIn("specificity", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.75)
        self.assertGreaterEqual(metrics["sensitivity_recall"], 0.75)


if __name__ == "__main__":
    unittest.main()
