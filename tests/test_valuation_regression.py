"""Unit tests for Ford Automotive Valuation Regression pipeline."""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "05-ford-automotive-valuation-regression"))

from valuation_pipeline import AutomotiveValuationPipeline
from train_valuation_model import generate_synthetic_vehicle_listings


class TestAutomotiveValuationPipeline(unittest.TestCase):
    """Test suite for vehicle valuation regression."""

    def setUp(self):
        self.pipeline = AutomotiveValuationPipeline()

    def test_price_monotonic_depreciation_with_age(self):
        """Verify older vehicles with identical specs have lower predicted prices."""
        specs_new = {"vehicle_age": 1.0, "mileage": 12000.0, "engine_size": 1.5, "mpg": 45.0}
        specs_old = {"vehicle_age": 8.0, "mileage": 12000.0, "engine_size": 1.5, "mpg": 45.0}

        price_new = self.pipeline.predict_price(specs_new)
        price_old = self.pipeline.predict_price(specs_old)
        self.assertGreater(price_new, price_old)
        self.assertGreater(price_new, 15000.0)

    def test_price_depreciation_with_mileage(self):
        """Verify high mileage vehicles have lower predicted prices."""
        specs_low_miles = {"vehicle_age": 3.0, "mileage": 15000.0, "engine_size": 1.5, "mpg": 45.0}
        specs_high_miles = {"vehicle_age": 3.0, "mileage": 95000.0, "engine_size": 1.5, "mpg": 45.0}

        price_low = self.pipeline.predict_price(specs_low_miles)
        price_high = self.pipeline.predict_price(specs_high_miles)
        self.assertGreater(price_low, price_high)

    def test_cohort_regression_evaluation(self):
        """Verify cohort evaluation returns valid positive metrics and high R2."""
        cohort = generate_synthetic_vehicle_listings(n_listings=100)
        self.assertEqual(len(cohort), 100)

        metrics = self.pipeline.evaluate_valuation_cohort(cohort)
        self.assertGreater(metrics["r2"], 0.70)
        self.assertGreater(metrics["mae"], 0.0)
        self.assertGreater(metrics["rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
