
# Random Forest - API Deployment Package
- Run locally: `pip install -r requirements.txt` then `uvicorn main:app --reload`
- Run with Docker: `docker build -t random_forest .` then `docker run -p 8000:8000 random_forest`
