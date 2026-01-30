from .celery_app import celery_app
from .train_models import run_automl_training
from .data_loader import load_tabular_data_from_bytes
import pandas as pd

@celery_app.task(bind=True)
def run_automl_task(self, file_content_bytes, filename, target_column, experiment_name, config, sensitive_feature):
    """
    Celery task that receives the full configuration from the frontend and passes
    it to the core training function.
    """
    # --- START OF FIX ---
    # The function signature now correctly includes the 'config' dictionary.
    
    self.update_state(state='PROGRESS', meta={'status': 'Loading data...'})
    df = load_tabular_data_from_bytes(file_content_bytes, filename)
    
    self.update_state(state='PROGRESS', meta={'status': 'Starting MLflow experiment...'})
    
    # Pass the config dictionary down to the core training function.
    parent_run_id = run_automl_training(df, target_column, experiment_name, config, sensitive_feature)
    
    # --- END OF FIX ---
    
    return {'status': 'COMPLETED', 'result': {'mlflow_run_id': parent_run_id}}