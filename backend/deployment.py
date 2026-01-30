import os
import joblib

def create_deployment_package(pipeline_path: str, serving_name: str, task_type: str):
    """
    Creates a folder with all the artifacts for a standalone FastAPI deployment.
    This folder can be zipped and sent to any cloud platform that supports Docker.

    Returns:
        str: The path to the created deployment directory.
    """
    deployment_dir = os.path.join('deployments', serving_name.replace(' ', '_').lower())
    os.makedirs(deployment_dir, exist_ok=True)
    
    # 1. Copy the pipeline model asset
    model_filename = "pipeline.pkl"
    destination_model_path = os.path.join(deployment_dir, model_filename)
    joblib.dump(joblib.load(pipeline_path), destination_model_path)

    # 2. Create the main.py for the FastAPI app
    main_py_content = f"""
import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Dict, Any

app = FastAPI(title="{serving_name} Deployment")

# Load the entire pipeline
pipeline = joblib.load('{model_filename}')

# Define the input data model using Pydantic
class PredictionInput(BaseModel):
    data: List[Dict[str, Any]]

@app.get("/")
def read_root():
    return {{"message": "Welcome to the {serving_name} model API."}}

@app.post("/predict")
def predict(payload: PredictionInput):
    try:
        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame(payload.data)
        
        # Ensure correct column order/types if necessary (advanced)
        
        # Make predictions
        predictions = pipeline.predict(df).tolist()
        
        if "{task_type}" == 'classification':
            try:
                # Predict probabilities for classification
                probabilities = pipeline.predict_proba(df).tolist()
                return {{"predictions": predictions, "probabilities": probabilities}}
            except AttributeError: # Some models might not have predict_proba
                return {{"predictions": predictions}}
        else:
            return {{"predictions": predictions}}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
    with open(os.path.join(deployment_dir, "main.py"), "w") as f:
        f.write(main_py_content)

    # 3. Create the requirements.txt for the deployment
    requirements_content = """
fastapi
uvicorn
scikit-learn
pandas
joblib
xgboost
"""
    with open(os.path.join(deployment_dir, "requirements.txt"), "w") as f:
        f.write(requirements_content)
        
    # 4. Create a simple Dockerfile for deployment
    dockerfile_content = """
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    with open(os.path.join(deployment_dir, "Dockerfile"), "w") as f:
        f.write(dockerfile_content)

    print(f"Deployment package created at: {deployment_dir}")
    return deployment_dir