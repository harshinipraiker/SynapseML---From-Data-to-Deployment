
import uvicorn; import joblib; import pandas as pd; from fastapi import FastAPI, HTTPException
from pydantic import BaseModel; from typing import List, Dict, Any
app = FastAPI(title="XGBoost API Deployment"); pipeline = joblib.load('pipeline.pkl')
class PredictionInput(BaseModel): data: List[Dict[str, Any]]
@app.get("/")
def read_root(): return {"message": "Welcome to the XGBoost model API."}
@app.post("/predict")
def predict(payload: PredictionInput):
    try:
        df = pd.DataFrame(payload.data); predictions = pipeline.predict(df).tolist()
        if "classification" == 'classification' and hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(df).tolist()
            return {"predictions": predictions, "probabilities": probabilities}
        else: return {"predictions": predictions}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=8000)
