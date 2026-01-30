# XGBoost Deployment
This folder contains a deployable FastAPI service.
## Run Locally
`pip install -r requirements.txt`
`uvicorn main:app --reload`
## Run with Docker
`docker build -t xgboost .`
`docker run -p 8000:8000 xgboost`
