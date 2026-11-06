"""Train regression models for automotive valuation and secondary market price forecasting."""

from typing import Dict, Any, List
import random
from valuation_pipeline import AutomotiveValuationPipeline

try:
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def generate_synthetic_vehicle_listings(n_listings: int = 500) -> List[Dict[str, float]]:
    """Synthesize representative secondary automotive market inventory."""
    random.seed(42)
    listings = []
    base_pipeline = AutomotiveValuationPipeline()

    for _ in range(n_listings):
        vehicle_age = random.uniform(0.5, 12.0)
        mileage = vehicle_age * random.uniform(7000.0, 14000.0) + random.uniform(0.0, 5000.0)
        engine_size = random.choice([1.0, 1.2, 1.5, 2.0, 2.3])
        mpg = random.uniform(32.0, 65.0)
        transmission_manual = float(random.choice([0, 1]))
        fuel_diesel = float(random.choice([0, 1]))

        specs = {
            "vehicle_age": round(vehicle_age, 1),
            "mileage": round(mileage, 0),
            "engine_size": engine_size,
            "mpg": round(mpg, 1),
            "transmission_manual": transmission_manual,
            "fuel_diesel": fuel_diesel,
        }
        # Add market variance / noise
        actual_price = base_pipeline.predict_price(specs) + random.gauss(0.0, 650.0)
        specs["price"] = round(max(actual_price, 1200.0), 2)
        listings.append(specs)

    return listings


def main():
    print("=" * 75)
    print("  FORD SECONDARY MARKET VEHICLE VALUATION & REGRESSION ENGINE")
    print("=" * 75)

    listings = generate_synthetic_vehicle_listings(n_listings=600)
    print(f"Generated {len(listings)} secondary automotive vehicle records.")

    pipeline = AutomotiveValuationPipeline()
    metrics = pipeline.evaluate_valuation_cohort(listings)
    print("\n[Native Calibrated Depreciation Model Performance]")
    print(f"  MAE (Mean Absolute Error):  ${metrics['mae']:,.2f}")
    print(f"  RMSE (Root Mean Sq Error):  ${metrics['rmse']:,.2f}")
    print(f"  R² Variance Explained:      {metrics['r2'] * 100:.2f}%")

    if SKLEARN_AVAILABLE:
        print("\n[Scikit-Learn Regression Benchmark Matrix]")
        features = pipeline.feature_names
        X = np.array([[row[f] for f in features] for row in listings])
        y = np.array([row["price"] for row in listings])

        split = int(0.8 * len(listings))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        models = {
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42),
        }

        print(f"{'Algorithm':<24} {'MAE (USD)':<12} {'RMSE (USD)':<13} {'R² Score':<10}")
        print("-" * 60)
        for name, reg in models.items():
            reg.fit(X_train, y_train)
            preds = reg.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            rmse = root_mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            print(f"{name:<24} ${mae:>8.2f}    ${rmse:>9.2f}     {r2:>7.4f}")

    print("=" * 75)


if __name__ == "__main__":
    main()
