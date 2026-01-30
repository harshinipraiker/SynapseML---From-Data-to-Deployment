import os
import io
import json
import traceback
import pandas as pd
import mlflow
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from celery.result import AsyncResult

from .celery_app import celery_app
from .tasks import run_automl_task
from .genai import get_basic_qna_chain

app = FastAPI(title="OpenWatsonX API Gateway")
mlflow.set_tracking_uri("sqlite:///mlflow.db")
celery_app.conf.update(result_extended=True)


@app.post("/submit_automl_job")
async def submit_automl_job(
    file: UploadFile = File(...), 
    target_column: str = Form(...), 
    experiment_name: str = Form(...), 
    config_str: str = Form(...),
    sensitive_feature: str = Form(None)
):
    try:
        file_content = await file.read()
        config = json.loads(config_str)
        task = run_automl_task.delay(
            file_content, file.filename, target_column, 
            experiment_name, config, sensitive_feature or ''
        )
        return JSONResponse({"task_id": task.id})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Failed to submit job: {e}"})


@app.get("/job_status/{task_id}")
async def get_job_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "state": task_result.state, "details": task_result.info}

    if task_result.state == 'SUCCESS' and task_result.result:
        try:
            run_id = task_result.result.get('result', {}).get('mlflow_run_id')
            if run_id:
                client = mlflow.tracking.MlflowClient()
                child_runs = client.search_runs(
                    experiment_ids=[mlflow.get_run(run_id).info.experiment_id], 
                    filter_string=f"tags.mlflow.parentRunId = '{run_id}'"
                )
                
                # Flatten the data structure for a clean frontend experience
                leaderboard_data = []
                for run in child_runs:
                    run_dict = {"run_id": run.info.run_id}
                    run_dict.update({f"metrics.{k}": v for k, v in run.data.metrics.items()})
                    run_dict.update({f"tags.{k}": v for k, v in run.data.tags.items()})
                    leaderboard_data.append(run_dict)

                response['leaderboard'] = leaderboard_data
            else:
                response['details'] = {'error': 'Task finished but no MLflow run ID was found.'}
        except Exception as e:
            traceback.print_exc()
            response['leaderboard'] = []
            response['details'] = {'error': f'Could not fetch MLflow results: {str(e)}'}
    
    return JSONResponse(response)


@app.post("/test_model")
async def test_model(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    experiment_name: str = Form(...)
):
    try:
        # Load the latest version of the registered model
        model_uri = f"models:/{experiment_name}_{model_name}/latest"
        print(f"[TESTER] Loading model from URI: {model_uri}")
        pipeline = mlflow.sklearn.load_model(model_uri)
        
        df_test = pd.read_csv(file.file)
        
        # Make predictions
        print(f"[TESTER] Making predictions on {df_test.shape[0]} rows.")
        df_test['predictions'] = pipeline.predict(df_test)
        
        # Also return probabilities if it's a classification model
        if hasattr(pipeline, "predict_proba"):
            try:
                prob_df = pd.DataFrame(pipeline.predict_proba(df_test), columns=[f"prob_class_{c}" for c in pipeline.classes_])
                df_test = pd.concat([df_test, prob_df], axis=1)
            except Exception as e:
                print(f"[TESTER WARNING] Could not get probabilities: {e}")

        return JSONResponse(df_test.to_dict(orient='records'))
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/run_details/{run_id}")
async def get_run_details(run_id: str):
    try:
        run = mlflow.get_run(run_id)
        return JSONResponse({
            "params": run.data.params, 
            "metrics": run.data.metrics, 
            "tags": run.data.tags,
            "info": { 
                "run_id": run.info.run_id, 
                "experiment_id": run.info.experiment_id, 
                "status": run.info.status,
                "start_time": str(run.info.start_time),
                "end_time": str(run.info.end_time)
            }
        })
    except Exception as e:
        return JSONResponse(status_code=404, content={"message": f"Run '{run_id}' not found: {str(e)}"})


@app.get("/run_artifacts/list/{run_id}")
async def list_run_artifacts(run_id: str):
    try:
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id=run_id, path="explainability")
        artifact_list = [{"path": art.path, "is_dir": art.is_dir, "size": art.file_size} for art in artifacts]
        return JSONResponse(artifact_list)
    except Exception as e:
        return JSONResponse(status_code=404, content={"message": f"Artifacts not found for run {run_id}: {str(e)}"})


@app.get("/run_artifacts/get/{run_id}")
async def get_run_artifact(run_id: str, path: str):
    try:
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path)
        media_type = "image/png" if path.lower().endswith(".png") else "application/octet-stream"
        return FileResponse(local_path, media_type=media_type, filename=os.path.basename(path))
    except Exception as e:
        return JSONResponse(status_code=404, content={"message": f"Artifact '{path}' not found: {str(e)}"})


@app.post("/genai/qna")
async def handle_qna(request_data: dict):
    question = request_data.get("question")
    if not question:
        return JSONResponse(status_code=400, content={"error": "Question not provided."})
    try:
        llm_chain = get_basic_qna_chain()
        if not llm_chain:
             return JSONResponse(status_code=500, content={"error": "LLM could not be initialized. Check API token."})
        answer = llm_chain.run(question)
        return JSONResponse({"answer": answer})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})