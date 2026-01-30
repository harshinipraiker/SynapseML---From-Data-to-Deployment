
import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Dict, Any

app = FastAPI(title="automl-run-20251106_XGBoost Deployment")

# Load the entire pipeline
pipeline = joblib.load('pipeline.pkl')

# Define the input data model using Pydantic
class PredictionInput(BaseModel):
    data: List[Dict[str, Any]]

@app.get("/")
def read_root():
    return {"message": "Welcome to the automl-run-20251106_XGBoost model API."}

@app.post("/predict")
def predict(payload: PredictionInput):
    try:
        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame(payload.data)
        
        # Ensure correct column order/types if necessary (advanced)
        
        # Make predictions
        predictions = pipeline.predict(df).tolist()
        
        if "regression" == 'classification':
            try:
                # Predict probabilities for classification
                probabilities = pipeline.predict_proba(df).tolist()
                return {"predictions": predictions, "probabilities": probabilities}
            except AttributeError: # Some models might not have predict_proba
                return {"predictions": predictions}
        else:
            return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
