"""Train and validate clinical cardiology risk models with Random Forest, XGBoost & Logistic Regression."""

from typing import Dict, Any, List
import random
from cardio_pipeline import ClinicalRiskClassifier

try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def generate_synthetic_clinical_cohort(n_patients: int = 500) -> List[Dict[str, float]]:
    """Synthesize representative clinical trial patient cohort based on Cleveland Heart Disease Schema."""
    random.seed(42)
    cohort = []
    for _ in range(n_patients):
        age = random.uniform(35.0, 78.0)
        resting_bp = random.uniform(94.0, 195.0)
        cholesterol = random.uniform(130.0, 360.0)
        max_heart_rate = random.uniform(85.0, 202.0)
        st_depression = random.uniform(0.0, 4.8)
        vessels_colored = random.choice([0.0, 1.0, 2.0, 3.0])
        chest_pain_type = random.choice([0.0, 1.0, 2.0, 3.0])
        exercise_angina = random.choice([0.0, 1.0])

        # Latent risk score
        latent = (
            0.045 * age
            + 0.018 * resting_bp
            + 0.012 * cholesterol
            - 0.022 * max_heart_rate
            + 0.850 * st_depression
            + 0.920 * vessels_colored
            + 0.650 * chest_pain_type
            + 0.780 * exercise_angina
            - 4.20
        )
        prob = 1.0 / (1.0 + (2.71828 ** (-latent)))
        target = 1.0 if prob > 0.52 else 0.0

        cohort.append({
            "age": round(age, 1),
            "resting_bp": round(resting_bp, 1),
            "cholesterol": round(cholesterol, 1),
            "max_heart_rate": round(max_heart_rate, 1),
            "st_depression": round(st_depression, 2),
            "vessels_colored": vessels_colored,
            "chest_pain_type": chest_pain_type,
            "exercise_angina": exercise_angina,
            "target": target,
        })
    return cohort


def main():
    print("=" * 75)
    print("  CARDIOVASCULAR CLINICAL RISK INTELLIGENCE & ML PREDICTIVE ENGINE")
    print("=" * 75)
    
    cohort = generate_synthetic_clinical_cohort(n_patients=600)
    print(f"Synthesized clinical cardiology cohort: {len(cohort)} patients")

    # Native Calibrated Classifier Evaluation
    classifier = ClinicalRiskClassifier()
    metrics = classifier.evaluate_batch(cohort)
    print("\n[Native Calibrated Risk Model Performance]")
    print(f"  Accuracy:            {metrics['accuracy'] * 100:.2f}%")
    print(f"  Clinical Sensitivity (Recall): {metrics['sensitivity_recall'] * 100:.2f}%")
    print(f"  Specificity:         {metrics['specificity'] * 100:.2f}%")
    print(f"  F1-Score:            {metrics['f1_score']:.4f}")

    if SKLEARN_AVAILABLE:
        print("\n[Scikit-Learn Ensemble Benchmark Matrix]")
        feature_keys = classifier.feature_names
        X = np.array([[p[k] for k in feature_keys] for p in cohort])
        y = np.array([p["target"] for p in cohort])

        split_idx = int(0.8 * len(cohort))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
        }

        print(f"{'Algorithm':<24} {'Accuracy':<10} {'Sensitivity':<13} {'ROC-AUC':<10}")
        print("-" * 60)
        for name, clf in models.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred)
            sens = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            print(f"{name:<24} {acc * 100:>6.2f}%    {sens * 100:>6.2f}%       {auc:>6.4f}")
    
    print("=" * 75)


if __name__ == "__main__":
    main()
