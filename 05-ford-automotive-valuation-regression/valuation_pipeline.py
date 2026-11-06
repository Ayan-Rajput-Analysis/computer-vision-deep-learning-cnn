"""Automotive Secondary Market Valuation & Price Regression Pipeline.

Addresses valuation uncertainty in secondary automotive markets by modeling historical
depreciation variables (mileage, engine displacement, transmission, fuel efficiency, age).
"""

from typing import Dict, Any, Tuple, List
import math


class AutomotiveValuationPipeline:
    """Valuation and price elasticity regression model."""

    def __init__(self):
        self.feature_names = [
            "vehicle_age",
            "mileage",
            "engine_size",
            "mpg",
            "transmission_manual",
            "fuel_diesel",
        ]
        # Calibrated regression coefficients for secondary Ford marketplace valuation
        self.base_price = 24500.0
        self.depreciation_rate_per_year = -1420.0
        self.depreciation_per_10k_miles = -680.0
        self.engine_size_premium = 2850.0
        self.efficiency_premium_per_mpg = 85.0
        self.manual_discount = -950.0
        self.diesel_premium = 450.0

    def predict_price(self, vehicle_specs: Dict[str, float]) -> float:
        """Estimate vehicle valuation in USD.

        Args:
            vehicle_specs: Dictionary containing vehicle attributes.

        Returns:
            Predicted vehicle market price (USD).
        """
        price = self.base_price
        price += self.depreciation_rate_per_year * float(vehicle_specs.get("vehicle_age", 3.0))
        price += self.depreciation_per_10k_miles * (float(vehicle_specs.get("mileage", 35000.0)) / 10000.0)
        price += self.engine_size_premium * (float(vehicle_specs.get("engine_size", 1.5)) - 1.0)
        price += self.efficiency_premium_per_mpg * (float(vehicle_specs.get("mpg", 45.0)) - 40.0)
        if vehicle_specs.get("transmission_manual", 0.0) == 1.0:
            price += self.manual_discount
        if vehicle_specs.get("fuel_diesel", 0.0) == 1.0:
            price += self.diesel_premium

        return max(round(price, 2), 1500.0)

    def evaluate_valuation_cohort(self, cohort: List[Dict[str, float]]) -> Dict[str, float]:
        """Compute regression evaluation metrics (MAE, RMSE, R-squared)."""
        if not cohort:
            return {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "n_samples": 0}

        y_true = [float(item["price"]) for item in cohort]
        y_pred = [self.predict_price(item) for item in cohort]

        n = len(y_true)
        mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n
        mse = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n
        rmse = math.sqrt(mse)

        mean_y = sum(y_true) / n
        ss_tot = sum((t - mean_y) ** 2 for t in y_true)
        ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
        r2 = 1.0 - (ss_res / max(ss_tot, 1e-6))

        return {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "n_samples": n,
        }
