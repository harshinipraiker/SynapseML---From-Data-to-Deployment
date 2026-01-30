import streamlit as st
import pandas as pd
import os
import numpy as np
import shap
import matplotlib.pyplot as plt
import graphviz
import uuid
import optuna
import joblib
import textwrap
import shutil
import io
import requests
from datetime import datetime
from streamlit_lottie import st_lottie

# --- IMPORTS for a REAL, ROBUST Model Training Pipeline ---
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score, f1_score

# CLASSIFICATION & REGRESSION MODELS
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC, SVR

# --- Page Config ---
st.set_page_config(layout="wide", page_title="SynapseML")

# --- NEW: Hyper-Animated Light Theme ---
st.markdown("""
<style>
    /* Keyframe Animations */
    @keyframes slideInUp {
        from { transform: translateY(50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 5px 0px rgba(0, 123, 255, 0.3); }
        50% { box-shadow: 0 0 20px 5px rgba(0, 123, 255, 0.5); }
        100% { box-shadow: 0 0 5px 0px rgba(0, 123, 255, 0.3); }
    }
    @keyframes gradientText {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes swipeIn {
        from { background-position: 100% 0; }
        to { background-position: 0 0; }
    }

    /* Base Styling */
    .stApp { background-color: #F0F4F8; }
    .main .block-container { padding: 2rem 5rem; }

    /* Animated Title */
    .animated-title {
        font-weight: 900;
        background: linear-gradient(90deg, #007BFF, #00BFFF, #007BFF);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientText 5s ease infinite;
    }
    
    /* Animated Section Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #111827;
        font-weight: 700;
    }
    h1:not(.animated-title), h2, h3 {
        background: linear-gradient(90deg, #111827 50%, #6b7280 100%);
        background-size: 200% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: swipeIn 1.5s ease-out;
    }

    /* Pulsing Call-to-Action Button */
    .stButton>button {
        border-radius: 10px; font-weight: bold; color: white; border: none; padding: 12px 24px;
        background-color: #007BFF;
        animation: pulseGlow 2.5s infinite;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 7px 20px 0 rgba(0, 123, 255, 0.45);
        background-color: #0069D9;
    }

    /* Animated "Card" Containers */
    [data-testid="stVerticalBlockBorderWrapper"], .st-expander {
        border: 1px solid #E5E7EB !important;
        border-radius: 15px !important;
        background-color: #FFFFFF;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease-in-out;
        animation: slideInUp 0.8s ease-out;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover, .st-expander:hover {
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.1);
        transform: translateY(-5px);
    }
    .st-expander header { font-size: 1.2em; font-weight: bold; color: #111827; }

</style>
""", unsafe_allow_html=True)


# --- Session State Initialization ---
keys_to_init = {
    'results_ready': False, 'results_data': None, 'task_type': None, 'dataframe': None,
    'config': None, 'leaderboard': None, 'best_model': None, 'deployment_package_zip_path': None
}
for key, value in keys_to_init.items():
    if key not in st.session_state: st.session_state[key] = value

# --- HELPER & BACKEND FUNCTIONS ---

@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

def clear_results_on_new_upload():
    for key in list(st.session_state.keys()):
        if not key.startswith("_"): del st.session_state[key]

def identify_task_type(y: pd.Series):
    if pd.api.types.is_float_dtype(y) or (pd.api.types.is_integer_dtype(y) and y.nunique() > 25): return 'regression'
    else: return 'classification'

def create_pipeline_flowchart(details, filename="flowchart"):
    # The backend functions remain the same. The UI is the focus of this update.
    dot = graphviz.Digraph(comment='Pipeline Flowchart')
    dot.attr('graph', rankdir='LR', bgcolor='transparent')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='#E9ECEF', color='#007BFF', penwidth='2')
    dot.attr('edge', color='#6C757D')
    dot.node('start', 'Input\nDataset', shape='ellipse', fillcolor='#D1E7DD', style='filled')
    pipeline_steps = f"Preprocessing\n(Imputer: {details['imputation']})\n(Scaler: {details['scaler']})"
    dot.node('preprocess', pipeline_steps)
    model_node_text = f"Model Training\n({details.get('model_name', 'N/A')})"
    if details.get('tuned'): model_node_text += "\n(Hyperparameter Tuned)"
    dot.node('model', model_node_text, fillcolor='#CFE2FF', style='filled')
    dot.node('eval', 'Evaluation\n(Metrics & SHAP)')
    dot.node('end', 'Deployment\nPackage', shape='ellipse', fillcolor='#F8D7DA', style='filled')
    dot.edge('start', 'preprocess'); dot.edge('preprocess', 'model'); dot.edge('model', 'eval'); dot.edge('eval', 'end')
    os.makedirs("temp_outputs", exist_ok=True); output_path = os.path.join("temp_outputs", filename)
    dot.render(output_path, format='png', cleanup=True); return f"{output_path}.png"

def generate_shap_plot(model, X_processed, feature_names, model_name, run_id):
    try:
        if isinstance(model, (RandomForestClassifier, RandomForestRegressor, XGBClassifier, XGBRegressor)): explainer = shap.TreeExplainer(model)
        else: explainer = shap.KernelExplainer(model.predict, shap.kmeans(X_processed, 10))
        shap_values = explainer.shap_values(X_processed); plt.figure()
        shap.summary_plot(shap_values, X_processed, feature_names=feature_names, show=False, plot_size="auto"); plt.tight_layout()
        os.makedirs("temp_outputs", exist_ok=True); shap_plot_path = os.path.join("temp_outputs", f"shap_{run_id}.png")
        plt.savefig(shap_plot_path); plt.close()
        return shap_plot_path
    except Exception as e:
        st.warning(f"Could not generate SHAP plot for {model_name}. Error: {e}")
        return None

def run_automl_pipeline(df: pd.DataFrame, target_column: str, config: dict):
    # This entire backend function remains unchanged.
    if config['numeric_imputation'] == 'skip' or config['categorical_imputation'] == 'skip': df.dropna(inplace=True)
    X = df.drop(target_column, axis=1); y = df[target_column]
    task_type = identify_task_type(y)
    if task_type == 'classification':
        le = LabelEncoder(); y_encoded = le.fit_transform(y); all_classes = le.classes_
    else: y_encoded = y
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
    numeric_pipeline_steps = []
    if config['numeric_imputation'] != 'skip': numeric_pipeline_steps.append(('imputer', SimpleImputer(strategy=config['numeric_imputation'])))
    if config['numeric_scaler'] == 'standard': numeric_pipeline_steps.append(('scaler', StandardScaler()))
    elif config['numeric_scaler'] == 'minmax': numeric_pipeline_steps.append(('scaler', MinMaxScaler()))
    elif config['numeric_scaler'] == 'robust': numeric_pipeline_steps.append(('scaler', RobustScaler()))
    categorical_pipeline_steps = []
    if config['categorical_imputation'] != 'skip': categorical_pipeline_steps.append(('imputer', SimpleImputer(strategy=config['categorical_imputation'])))
    categorical_pipeline_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore')))
    preprocessor = ColumnTransformer(transformers=[('num', Pipeline(steps=numeric_pipeline_steps), numeric_features), ('cat', Pipeline(steps=categorical_pipeline_steps), categorical_features)])
    all_models = {'classification': {"Logistic Regression": LogisticRegression(max_iter=1000), "Random Forest": RandomForestClassifier(random_state=42), "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), "SVM": SVC(probability=True, random_state=42)}, 'regression': {"Linear Regression": LinearRegression(), "Random Forest": RandomForestRegressor(random_state=42), "XGBoost": XGBRegressor(random_state=42), "SVM": SVR()}}
    models_to_train = {name: model for name, model in all_models[task_type].items() if name in config['models']}
    if not models_to_train: st.error("No valid models were selected for the detected task."); return None, None
    results = {}; progress_bar = st.progress(0, text="Initializing AutoML run...")
    for i, (name, model) in enumerate(models_to_train.items()):
        run_id = uuid.uuid4().hex[:8]; best_params = {}; is_tuned = False
        if config['enable_tuning'] and name in ["Random Forest", "XGBoost", "SVM"]:
            is_tuned = True
            progress_bar.progress((i + 0.1) / len(models_to_train), text=f"Tuning {name} with Optuna...")
            def objective(trial, model_obj=model, model_name=name):
                if model_name == "Random Forest": params = {'n_estimators': trial.suggest_int('n_estimators', 50, 300), 'max_depth': trial.suggest_int('max_depth', 5, 50, log=True)}
                elif model_name == "SVM": params = {'C': trial.suggest_float('C', 1e-2, 1e2, log=True), 'kernel': trial.suggest_categorical('kernel', ['rbf'])}
                else: params = {'n_estimators': trial.suggest_int('n_estimators', 50, 300), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)}
                temp_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model_obj.set_params(**params))])
                score = cross_val_score(temp_pipeline, X_train, y_train, n_jobs=-1, cv=3, scoring='roc_auc' if task_type == 'classification' else 'r2').mean(); return score
            study = optuna.create_study(direction='maximize'); study.optimize(objective, n_trials=config['n_trials']); best_params = study.best_params
        progress_bar.progress((i + 0.5) / len(models_to_train), text=f"Training final {name} model...")
        final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model.set_params(**best_params))]); final_pipeline.fit(X_train, y_train)
        metrics = {'Accuracy': '-', 'F1 Score': '-', 'ROC AUC': '-', 'RMSE': '-', 'R2 Score': '-'}
        if task_type == 'classification':
            preds = final_pipeline.predict(X_test); metrics['Accuracy'] = f"{accuracy_score(y_test, preds):.4f}"; metrics['F1 Score'] = f"{f1_score(y_test, preds, average='weighted'):.4f}"
            if hasattr(final_pipeline, "predict_proba"):
                proba = final_pipeline.predict_proba(X_test); n_classes = len(all_classes)
                if n_classes == 2: metrics['ROC AUC'] = f"{roc_auc_score(y_test, proba[:, 1]):.4f}"
                else: metrics['ROC AUC'] = f"{roc_auc_score(y_test, proba, multi_class='ovr', labels=np.arange(n_classes)):.4f}"
            else: metrics['ROC AUC'] = "N/A"
        else:
            preds = final_pipeline.predict(X_test); metrics['R2 Score'] = f"{r2_score(y_test, preds):.4f}"; metrics['RMSE'] = f"{np.sqrt(mean_squared_error(y_test, preds)):.4f}"
        X_test_processed = final_pipeline.named_steps['preprocessor'].transform(X_test)
        feature_names_out = final_pipeline.named_steps['preprocessor'].get_feature_names_out()
        shap_path = generate_shap_plot(final_pipeline.named_steps['model'], X_test_processed, feature_names_out, name, run_id)
        results[run_id] = {'model_name': name, 'metrics': metrics, 'params': best_params, 'shap_plot_path': shap_path, 'tuned': is_tuned, 'pipeline': final_pipeline}
        progress_bar.progress((i + 1) / len(models_to_train), text=f"Completed training for {name}")
    return results, task_type

def generate_deployment_package(pipeline: object, model_name: str, task_type: str) -> str:
    # This entire backend function remains unchanged.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S"); safe_model_name = model_name.replace(" ", "_").lower()
    deployment_dir = os.path.join("DEPLOYMENT_OUTPUT", f"{safe_model_name}_{timestamp}")
    if os.path.exists(deployment_dir): shutil.rmtree(deployment_dir)
    os.makedirs(deployment_dir, exist_ok=True); joblib.dump(pipeline, os.path.join(deployment_dir, "pipeline.pkl"))
    main_py_content = f"""
import uvicorn; import joblib; import pandas as pd; from fastapi import FastAPI, HTTPException
from pydantic import BaseModel; from typing import List, Dict, Any
app = FastAPI(title="{model_name} API Deployment"); pipeline = joblib.load('pipeline.pkl')
class PredictionInput(BaseModel): data: List[Dict[str, Any]]
@app.get("/")
def read_root(): return {{"message": "Welcome to the {model_name} model API."}}
@app.post("/predict")
def predict(payload: PredictionInput):
    try:
        df = pd.DataFrame(payload.data); predictions = pipeline.predict(df).tolist()
        if "{task_type}" == 'classification' and hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(df).tolist()
            return {{"predictions": predictions, "probabilities": probabilities}}
        else: return {{"predictions": predictions}}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=8000)
"""
    with open(os.path.join(deployment_dir, "main.py"), "w") as f: f.write(textwrap.dedent(main_py_content))
    with open(os.path.join(deployment_dir, "requirements.txt"), "w") as f: f.write("fastapi\nuvicorn\npandas\nscikit-learn\njoblib\nxgboost\nnumpy\n")
    dockerfile_content = """
FROM python:3.9-slim
WORKDIR /app; COPY . .; RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000; CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    with open(os.path.join(deployment_dir, "Dockerfile"), "w") as f: f.write(textwrap.dedent(dockerfile_content))
    readme_content = f"""
# {model_name} - API Deployment Package
- Run locally: `pip install -r requirements.txt` then `uvicorn main:app --reload`
- Run with Docker: `docker build -t {safe_model_name} .` then `docker run -p 8000:8000 {safe_model_name}`
"""
    with open(os.path.join(deployment_dir, "README.md"), "w") as f: f.write(textwrap.dedent(readme_content))
    shutil.make_archive(deployment_dir, 'zip', deployment_dir); return f"{deployment_dir}.zip"

# --- >>> START OF THE STREAMLIT UI <<< ---

lottie_header_url = "https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json"
lottie_header_json = load_lottieurl(lottie_header_url)
lottie_loading_url = "https://assets8.lottiefiles.com/packages/lf20_1LhTE2.json"
lottie_loading_json = load_lottieurl(lottie_loading_url)

h_col1, h_col2 = st.columns([1, 6])
with h_col1:
    if lottie_header_json:
        st_lottie(lottie_header_json, height=120, width=120, speed=1, loop=True, quality='high')
with h_col2:
    st.markdown("<h1 class='animated-title'>SynapseML</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #A0AEC0; font-weight: 400;'>An Automated Machine Learning Workbench</h4>", unsafe_allow_html=True)

st.divider()
uploaded_file = st.file_uploader("Upload your CSV dataset to begin", type="csv", on_change=clear_results_on_new_upload)

if uploaded_file:
    st.session_state.dataframe = pd.read_csv(uploaded_file)
    with st.container(border=True):
        st.header(" 1. Configure Preprocessing & Models")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Cleaning")
            numeric_imputation = st.selectbox("Numeric Missing Values", options=['median', 'mean', 'most_frequent', 'constant', 'skip'], index=0)
            categorical_imputation = st.selectbox("Categorical Missing Values", options=['most_frequent', 'constant', 'skip'], index=0)
        with col2:
            st.subheader("Scaling & Model Selection")
            numeric_scaler = st.selectbox("Numeric Scaler", options=['standard', 'minmax', 'robust', 'none'], index=0)
            model_options = ["Linear Regression", "Logistic Regression", "SVM", "Random Forest", "XGBoost"]
            selected_models = st.multiselect("Models to Train", model_options, default=model_options)

        with st.expander(" Hyperparameter Tuning"):
            enable_tuning = st.checkbox("Enable Optuna Tuning for Tree-Based Models & SVM", False)
            tuning_iterations = 10
            if enable_tuning:
                tuning_iterations = st.slider("Number of Tuning Iterations (Trials)", 2, 50, 10, help="More trials can find better models but takes longer.")
    
    with st.container(border=True):
        st.header(" 2. Define Target & Start Training")
        df_columns = st.session_state.dataframe.columns.tolist()
        target_column = st.selectbox("Select Target Column", df_columns, index=len(df_columns)-1)
        if target_column:
            task_type_display = identify_task_type(st.session_state.dataframe[target_column])
            st.info(f"Detected Task: **{task_type_display.title()}**")
        sensitive_features_options = ["None"] + [col for col in df_columns if col != target_column]
        sensitive_feature = st.selectbox("Select Sensitive Feature (for fairness)", sensitive_features_options)
        
        st.divider()
        if st.button(" Launch AutoML Engine", use_container_width=True, type="primary"):
            config = {'numeric_imputation': numeric_imputation, 'categorical_imputation': categorical_imputation, 'numeric_scaler': numeric_scaler, 'models': selected_models, 'enable_tuning': enable_tuning, 'n_trials': tuning_iterations}
            st.session_state.config = config
            
            loading_placeholder = st.empty()
            with loading_placeholder.container():
                if lottie_loading_json:
                    st_lottie(lottie_loading_json, height=300, speed=1, loop=True, quality='high')
                st.write("### Running AutoML pipeline... This may take a few moments.")
            
            results, task_type = run_automl_pipeline(st.session_state.dataframe, target_column, config)
            loading_placeholder.empty()
            
            if results:
                primary_metric = "R2 Score" if task_type == 'regression' else "ROC AUC"
                def safe_float(value):
                    try: return float(value)
                    except (ValueError, TypeError): return -999.0
                sorted_runs = sorted(results.items(), key=lambda item: safe_float(item[1]['metrics'].get(primary_metric, -1)), reverse=True)
                best_model_name = sorted_runs[0][1]['model_name']
                st.session_state.update({'results_data': results, 'task_type': task_type, 'results_ready': True, 'best_model': best_model_name})
                st.rerun()

if st.session_state.results_ready:
    with st.container(border=True):
        st.header("3. Results & Deployment")
        all_results = st.session_state.results_data; task_type = st.session_state.task_type
        primary_metric = "ROC AUC" if task_type == 'classification' else "R2 Score"
        
        def safe_float(value):
            try: return float(value)
            except (ValueError, TypeError): return -999.0
        
        best_run_data = max(all_results.values(), key=lambda x: safe_float(x['metrics'].get(primary_metric, -1)))
        st.subheader("Best Overall Model")
        st.metric(label=best_run_data['model_name'], value=f"{best_run_data['metrics'].get(primary_metric, 'N/A')}", help=f"Based on {primary_metric}")

        tab1, tab2, tab3 = st.tabs(["Leaderboard", "Detailed Analysis", "Deployment"])

        with tab1:
            # UI code for tabs remains the same, but inherits the new styling
            pipeline_runs_data = []; tuned_runs_data = []
            for run_id, data in all_results.items():
                run_info = {'Model': data['model_name'], **data['metrics']}
                if data['tuned']:
                    params_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in data['params'].items()])
                    run_info['Best Parameters'] = params_str; tuned_runs_data.append(run_info)
                else: pipeline_runs_data.append(run_info)
            if pipeline_runs_data:
                st.subheader("Pipeline Leaderboard")
                pipeline_df = pd.DataFrame(pipeline_runs_data); pipeline_df['sort_metric'] = pipeline_df[primary_metric].apply(safe_float)
                pipeline_df = pipeline_df.sort_values(by='sort_metric', ascending=False).drop(columns=['sort_metric']).reset_index(drop=True)
                pipeline_df.insert(0, 'Rank', pipeline_df.index + 1)
                display_cols = ['Rank', 'Model', 'Accuracy', 'F1 Score', 'ROC AUC'] if task_type == 'classification' else ['Rank', 'Model', 'R2 Score', 'RMSE']
                st.dataframe(pipeline_df[[col for col in display_cols if col in pipeline_df.columns]], use_container_width=True)
            if tuned_runs_data:
                st.subheader("Optuna Tuned Leaderboard")
                tuned_df = pd.DataFrame(tuned_runs_data); tuned_df['sort_metric'] = tuned_df[primary_metric].apply(safe_float)
                tuned_df = tuned_df.sort_values(by='sort_metric', ascending=False).drop(columns=['sort_metric']).reset_index(drop=True)
                tuned_df.insert(0, 'Rank', tuned_df.index + 1)
                display_cols = ['Rank', 'Model', 'Best Parameters', 'Accuracy', 'F1 Score', 'ROC AUC'] if task_type == 'classification' else ['Rank', 'Model', 'Best Parameters', 'R2 Score', 'RMSE']
                st.dataframe(tuned_df[[col for col in display_cols if col in tuned_df.columns]], use_container_width=True)

        with tab2:
            st.subheader("Individual Run Analysis")
            sorted_runs = sorted(st.session_state.results_data.items(), key=lambda item: safe_float(item[1]['metrics'].get(primary_metric, -1)), reverse=True)
            for run_id, data in sorted_runs:
                with st.expander(f"🔬 **{data['model_name']}** (View Details)"):
                    st.subheader("Pipeline Flowchart")
                    flowchart_details = {'model_name': data['model_name'], 'tuned': data['tuned'], 'imputation': st.session_state.config['numeric_imputation'], 'scaler': st.session_state.config['numeric_scaler']}
                    flowchart_path = create_pipeline_flowchart(flowchart_details, f"flowchart_{run_id}")
                    st.image(flowchart_path)
                    st.subheader("Performance Metrics"); st.json({k: v for k, v in data['metrics'].items() if v != '-'})
                    st.subheader("Best Hyperparameters (from Optuna)")
                    if data['tuned']:
                        params_df = pd.DataFrame(data['params'].items(), columns=['Hyperparameter', 'Optimal Value']); st.table(params_df)
                    else: st.info("Hyperparameter tuning was not enabled for this model.")
                    st.subheader("Explainability (SHAP Plot)")
                    if data['shap_plot_path']: st.image(data['shap_plot_path'])
                    else: st.warning("SHAP plot could not be generated for this model.")
        
        with tab3:
            st.subheader("Generate Deployment Package")
            best_run_id, best_run_data = sorted(all_results.items(), key=lambda item: safe_float(item[1]['metrics'].get(primary_metric, -1)), reverse=True)[0]
            st.info(f"The best performing model, **{best_run_data['model_name']}**, has been selected for deployment.")
            if st.button(f"Generate Package for {best_run_data['model_name']}", use_container_width=True):
                with st.spinner("Generating professional deployment package..."):
                    zip_path = generate_deployment_package(best_run_data['pipeline'], best_run_data['model_name'], st.session_state.task_type)
                    st.session_state.deployment_package_zip_path = zip_path
            
            if st.session_state.deployment_package_zip_path:
                zip_path = st.session_state.deployment_package_zip_path
                st.success("Deployment package created successfully!")
                with open(zip_path, "rb") as f:
                    st.download_button(label=" Download Deployment Package", data=f, file_name=os.path.basename(zip_path), mime="application/zip", use_container_width=True, type="primary")
else:
    lottie_landing_url = "https://assets9.lottiefiles.com/packages/lf20_x9puxefh.json"
    lottie_landing_json = load_lottieurl(lottie_landing_url)
    if lottie_landing_json:
        st_lottie(lottie_landing_json, height=400, speed=1, loop=True, quality='high')
    st.info("Please upload a CSV file to begin your AutoML journey.")