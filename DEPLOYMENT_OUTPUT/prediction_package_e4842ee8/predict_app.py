
import streamlit as st; import pandas as pd; import joblib
try:
    pipeline = joblib.load("pipeline.pkl"); metadata = joblib.load("metadata.pkl")
    numeric_cols, all_features = metadata['numeric'], metadata['all_features']
except FileNotFoundError: st.error("Model files not found!"); st.stop()
st.set_page_config(page_title="Prediction App", layout="wide"); st.title("🚀 Deployed Model for Predictions")
st.write(f"This app uses the *'{'XGBoost'}*' model."); st.header("Enter Input Data for a Single Prediction")
input_data = {}; cols = st.columns(2)
for i, feature in enumerate(all_features):
    col = cols[i % 2]
    if feature in numeric_cols: input_data[feature] = col.number_input(f"'{feature}'", value=0.0, format="%.4f")
    else: input_data[feature] = col.text_input(f"'{feature}'", value="default_value")
if st.button("🔮 Make Prediction", type="primary", use_container_width=True):
    input_df = pd.DataFrame([input_data])
    try:
        prediction = pipeline.predict(input_df)
        probability = pipeline.predict_proba(input_df) if hasattr(pipeline, 'predict_proba') else None
        st.success("#### Prediction Result:"); st.metric(label="Predicted Value", value=f"{prediction[0]}")
        if probability is not None:
            st.write("Prediction Probabilities:"); st.dataframe(pd.DataFrame(probability, columns=[f'prob_class_{i}' for i in range(probability.shape[1])]))
    except Exception as e: st.error(f"An error occurred: {e}")
