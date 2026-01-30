
import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline and metadata
try:
    pipeline = joblib.load("pipeline.pkl")
    metadata = joblib.load("metadata.pkl")
    numeric_cols = metadata['numeric']
    all_features = metadata['all_features']
except FileNotFoundError:
    st.error("Model files not found! Make sure 'pipeline.pkl' and 'metadata.pkl' are in the same directory.")
    st.stop()

st.set_page_config(page_title="Prediction App", layout="wide")
st.title("🚀 Deployed Model for Predictions")
st.write(f"This app uses the **'{'XGBoost'}**' model.")

st.header("Enter Input Data for a Single Prediction")
input_data = {}
# Create two columns for a cleaner layout
cols = st.columns(2)

for i, feature in enumerate(all_features):
    col_container = cols[i % 2]
    if feature in numeric_cols:
        input_data[feature] = col_container.number_input(f"Enter value for '{feature}'", value=0.0, format="%.4f")
    else:
        input_data[feature] = col_container.text_input(f"Enter value for '{feature}'", value="default_value")

if st.button("🔮 Make Prediction", type="primary", use_container_width=True):
    input_df = pd.DataFrame([input_data])
    try:
        prediction = pipeline.predict(input_df)
        probability = pipeline.predict_proba(input_df) if hasattr(pipeline, 'predict_proba') else None

        st.success("#### Prediction Result:")
        st.metric(label="Predicted Value", value=f"{prediction[0]:.4f}")
        if probability is not None:
            st.write("Prediction Probabilities:")
            st.dataframe(pd.DataFrame(probability, columns=[f'prob_class_{i}' for i in range(probability.shape[1])]))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
