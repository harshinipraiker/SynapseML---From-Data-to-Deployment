# backend/train_models.py

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import mlflow
import os
import uuid
import traceback

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC, SVR

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_squared_error

from .task_identifier import identify_task
from .pipeline_viz import generate_lifecycle_flow

# --- SIMPLIFIED AND MORE ROBUST SHAP FUNCTION ---
def generate_shap_summary(pipeline, X_train, shap_path):
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']
    model_type = type(model)._name_
    print(f"--- [SHAP] Attempting SHAP for model: {model_type} ---")

    try:
        X_train_sample = X_train.head(100)
        X_train_transformed = preprocessor.transform(X_train_sample)
        
        if isinstance(X_train_transformed, np.ndarray):
            X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=[f'f_{i}' for i in range(X_train_transformed.shape[1])])
        else: # Handle sparse matrix from one-hot encoder if not dense
            X_train_transformed_df = pd.DataFrame(X_train_transformed.toarray(), columns=[f'f_{i}' for i in range(X_train_transformed.shape[1])])


        if model_type in ['RandomForestClassifier', 'RandomForestRegressor', 'XGBClassifier', 'XGBRegressor']:
            explainer = shap.TreeExplainer(model)
        else: 
            background_data = shap.kmeans(preprocessor.transform(X_train.head(50)), 10)
            predict_fn = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
            explainer = shap.KernelExplainer(predict_fn, background_data)

        shap_values = explainer.shap_values(X_train_transformed_df)
        
        plt.figure()
        plot_values = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values
        
        shap.summary_plot(plot_values, X_train_transformed_df, show=False)
        plt.tight_layout()
        plt.savefig(shap_path)
        plt.close()
        
        print(f"--- [SHAP] Summary plot saved successfully ---")
        return shap_path
        
    except Exception as e:
        print(f"!!!!!!!!!! [SHAP ERROR] for {model_type} !!!!!!!!!!")
        traceback.print_exc()
        return None

# --- MAIN TRAINING FUNCTION WITH MAX ROBUSTNESS ---
def run_automl_training(df: pd.DataFrame, target_column: str, experiment_name: str, config: dict, sensitive_feature: str = None):
    print(f"--- [WORKER] AutoML Job Started with Config: {config} ---")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    if config.get('numeric_imputation') == 'skip': 
        df.dropna(inplace=True)

    task_type = identify_task(df, target_column)
    
    y = df[target_column]
    X = df.drop(columns=[target_column])

    if task_type == 'classification': 
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    numeric_features = X.select_dtypes(include=np.number).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    cat_steps = [
        ('imputer', SimpleImputer(strategy='most_frequent' if config.get('categorical_imputation') == 'most_frequent' else 'constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]
    num_imputer_step = []
    if config.get('numeric_imputation') == 'median':
        num_imputer_step.append(('imputer', SimpleImputer(strategy='median')))

    all_models = {
        "classification": {
            "Logistic Regression": (LogisticRegression(max_iter=2000), {}),
            "Random Forest": (RandomForestClassifier(random_state=42), {'model__n_estimators': [100]}),
            "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {'model__n_estimators': [100]}),
            "SVM": (SVC(probability=True, random_state=42), {'model__C': [1.0]})
        },
        "regression": {
            "Linear Regression": (LinearRegression(), {}),
            "Random Forest": (RandomForestRegressor(random_state=42), {'model__n_estimators': [100]}),
            "XGBoost": (XGBRegressor(random_state=42), {'model__n_estimators': [100]}),
            "SVM": (SVR(), {'model__C': [1.0]})
        }
    }
    
    models_to_run = {name: model_info for name, model_info in all_models[task_type].items() if name in config.get('models', [])}

    with mlflow.start_run(run_name="AutoML_Parent_Run") as parent_run:
        mlflow.log_params(config)
        
        for name, (model, params) in models_to_run.items():
            child_run = None
            try:
                with mlflow.start_run(run_name=name, nested=True) as child_run:
                    run_id = child_run.info.run_id
                    print(f"\n[WORKER] Starting pipeline: {name}, Run ID: {run_id}")

                    numeric_steps = list(num_imputer_step) 
                    needs_scaling = name in ["Logistic Regression", "Linear Regression", "SVM"]
                    
                    if config.get('scaler') == 'standard' or needs_scaling:
                        print(f"[WORKER] Applying StandardScaler for {name}.")
                        numeric_steps.append(('scaler', StandardScaler()))
                    elif config.get('scaler') == 'minmax':
                         numeric_steps.append(('scaler', MinMaxScaler()))

                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', Pipeline(steps=numeric_steps) if numeric_steps else 'passthrough', numeric_features),
                            ('cat', Pipeline(steps=cat_steps), categorical_features)
                        ], remainder='passthrough'
                    )

                    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
                    
                    if params:
                        search = RandomizedSearchCV(pipeline, params, n_iter=1, cv=2, random_state=42)
                        search.fit(X_train, y_train)
                        best_pipeline = search.best_estimator_
                        mlflow.log_params(search.best_params_)
                    else:
                        best_pipeline = pipeline
                        best_pipeline.fit(X_train, y_train)
                    print(f"[WORKER] Fitting complete for {name}.")

                    # --- MODIFIED CODE BLOCK ---
                    # Generate and log the lifecycle flowchart visualization
                    print(f"[WORKER] Generating lifecycle flowchart for {name}...")
                    os.makedirs("reports", exist_ok=True)
                    flowchart_path = os.path.join("reports", f"flowchart_{run_id}.png")
                    
                    # The if statement handles the case where flowchart generation might fail
                    if generate_lifecycle_flow(best_pipeline, name, flowchart_path):
                        mlflow.log_artifact(flowchart_path, artifact_path="pipeline_visualization")
                        print(f"[WORKER] Flowchart logged to MLflow for {name}.")
                    # --- END OF MODIFIED CODE BLOCK ---

                    metrics = {}
                    try:
                        y_pred = best_pipeline.predict(X_test)
                        if task_type == 'classification':
                            if hasattr(best_pipeline, "predict_proba"):
                                metrics['ROC AUC'] = roc_auc_score(y_test, best_pipeline.predict_proba(X_test)[:, 1])
                            metrics['Accuracy'] = accuracy_score(y_test, y_pred)
                        else: 
                            metrics['R2 Score'] = r2_score(y_test, y_pred)
                            metrics['RMSE'] = np.sqrt(mean_squared_error(y_test, y_pred))
                        
                        if metrics:
                            print(f"[WORKER] Logging metrics for {name}: {metrics}")
                            mlflow.log_metrics(metrics)
                    except Exception as e:
                        print(f"!!!!!!!!!! [WORKER] METRIC ERROR for {name}: {e} !!!!!!!!!!")
                        mlflow.log_param("metric_error", str(e))

                    print(f"[WORKER] Logging model for {name}...")
                    mlflow.sklearn.log_model(sk_model=best_pipeline, artifact_path=name.replace(' ', ''), registered_model_name=f"{experiment_name}{name}")

                    os.makedirs("reports", exist_ok=True)
                    shap_path = os.path.join("reports", f"shap_{run_id}.png")
                    if generate_shap_summary(best_pipeline, X_train, shap_path):
                        mlflow.log_artifact(shap_path, artifact_path="explainability")
                        mlflow.set_tag("shap_plot_artifact", f"explainability/{os.path.basename(shap_path)}")

                    print(f"[WORKER] ✅ Completed pipeline: {name}")

            except Exception as e:
                print(f"!!!!!!!!!! [WORKER] ❌ FATAL ERROR in pipeline {name} !!!!!!!!!!")
                traceback.print_exc()
                if child_run: mlflow.log_param("fatal_error", str(e))

    print("--- [WORKER] AutoML Training Job Finished ---")
    return parent_run.info.run_id