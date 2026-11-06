"""Train gradient-boosted delivery risk predictive models with XGBoost & Scikit-Learn."""

from typing import Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pipeline import SupplyChainFeaturePipeline


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Calculate core classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def main():
    pipeline = SupplyChainFeaturePipeline()
    print("[Supply Chain Pipeline] Generating and processing data...")
    df = pipeline.create_synthetic_supply_chain_data(n_rows=2000)
    X_train, X_test, y_train, y_test = pipeline.fit_transform(df)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42),
    }

    try:
        import xgboost as xgb
        models["XGBoost (Optimized)"] = xgb.XGBClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42, eval_metric="logloss"
        )
    except ImportError:
        pass

    print("\n" + "=" * 70)
    print(f"{'Model':<24} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} {'ROC-AUC':<10}")
    print("=" * 70)

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        res = evaluate_model(y_test, y_pred, y_prob)
        print(
            f"{name:<24} {res['accuracy'] * 100:>6.2f}%    "
            f"{res['precision'] * 100:>6.2f}%     "
            f"{res['recall'] * 100:>6.2f}%   "
            f"{res['roc_auc']:>6.4f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
