
# Random Forest - API Deployment Package
This package contains a ready-to-deploy FastAPI service for the `Random Forest` model.
## How to Run Locally
1. `pip install -r requirements.txt`
2. `uvicorn main:app --reload`
## How to Run with Docker
1. `docker build -t random_forest .`
2. `docker run -p 8000:8000 random_forest`
