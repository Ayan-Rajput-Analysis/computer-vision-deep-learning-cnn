"""Unit tests for supply chain predictive pipeline."""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "03-supply-chain-predictive-analytics"))

try:
    import numpy as np
    import pandas as pd
    from pipeline import SupplyChainFeaturePipeline
    DATA_DEPS_AVAILABLE = True
except ImportError:
    DATA_DEPS_AVAILABLE = False


class TestPredictivePipeline(unittest.TestCase):
    """Test suite for supply chain data transformation pipeline."""

    def test_supply_chain_pipeline_split(self):
        """Verify data pipeline transforms features and generates consistent train/test shapes."""
        if not DATA_DEPS_AVAILABLE:
            self.skipTest("Data dependencies (numpy/pandas/sklearn) not installed locally")
        pipeline = SupplyChainFeaturePipeline()
        df = pipeline.create_synthetic_supply_chain_data(n_rows=100)
        self.assertEqual(len(df), 100)
        self.assertIn("Late_delivery_risk", df.columns)

        X_train, X_test, y_train, y_test = pipeline.fit_transform(df)
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)
        self.assertEqual(X_train.shape[1], len(pipeline.feature_names))
        self.assertFalse(np.isnan(X_train).any())


if __name__ == "__main__":
    unittest.main()
