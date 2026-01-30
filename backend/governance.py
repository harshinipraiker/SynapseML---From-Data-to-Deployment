from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from datetime import datetime
import os
import joblib

def evaluate_fairness(pipeline, X_test, y_test, sensitive_features):
    """Calculates fairness metrics."""
    y_pred = pipeline.predict(X_test)
    return {
        "demographic_parity_difference": demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_features),
        "equalized_odds_difference": equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_features),
    }

def get_versioned_path(model_name: str, experiment_name: str):
    """Generates a versioned model name for saving."""
    version = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_model_name = model_name.replace(' ', '_')
    safe_exp_name = experiment_name.replace(' ', '_')
    filename = f"{safe_exp_name}_{safe_model_name}_{version}"
    return filename