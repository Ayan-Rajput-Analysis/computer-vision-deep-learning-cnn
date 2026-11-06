"""Enterprise Data Pipeline & Feature Engineering for Supply Chain Risk Prediction."""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


class SupplyChainFeaturePipeline:
    """End-to-end transformation pipeline for order telemetry and delivery risk modeling."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.feature_names: List[str] = []

    def create_synthetic_supply_chain_data(self, n_rows: int = 1000) -> pd.DataFrame:
        """Generates synthetic supply chain dataset matching DataCo schema."""
        rng = np.random.default_rng(42)

        shipping_modes = ["Standard Class", "First Class", "Second Class", "Same Day"]
        order_regions = ["Western Europe", "Central America", "South America", "North America", "Southeast Asia"]
        categories = ["Cardio Equipment", "Cleats", "Shop By Sport", "Women's Apparel", "Electronics"]

        days_real = rng.integers(1, 7, size=n_rows)
        days_scheduled = rng.integers(1, 5, size=n_rows)
        # Late delivery if real shipping days exceed scheduled days
        late_risk = (days_real > days_scheduled).astype(int)

        df = pd.DataFrame(
            {
                "Days_for_shipping_real": days_real,
                "Days_for_shipment_scheduled": days_scheduled,
                "Benefit_per_order": rng.normal(30.0, 15.0, size=n_rows).round(2),
                "Sales_per_customer": rng.uniform(50.0, 450.0, size=n_rows).round(2),
                "Order_Item_Discount_Rate": rng.uniform(0.0, 0.25, size=n_rows).round(2),
                "Order_Item_Quantity": rng.integers(1, 6, size=n_rows),
                "Shipping_Mode": rng.choice(shipping_modes, size=n_rows),
                "Order_Region": rng.choice(order_regions, size=n_rows),
                "Category_Name": rng.choice(categories, size=n_rows),
                "Late_delivery_risk": late_risk,
            }
        )
        return df

    def fit_transform(
        self, df: pd.DataFrame, target_col: str = "Late_delivery_risk"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Performs stratified train/test split, encoding, and scaling."""
        X = df.drop(columns=[target_col])
        y = df[target_col].values

        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        num_cols = X.select_dtypes(include=["number"]).columns.tolist()

        X_encoded_cat = self.encoder.fit_transform(X[cat_cols])
        X_scaled_num = self.scaler.fit_transform(X[num_cols])

        X_processed = np.hstack([X_scaled_num, X_encoded_cat])
        self.feature_names = num_cols + cat_cols

        return train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
