# register model

import json
import mlflow
import logging
import os
import dagshub

# Set up MLflow tracking URI
# dagshub.init(repo_owner='Rohanpatil4600', repo_name='YT_comment', mlflow=True)
# mlflow.set_tracking_uri("https://dagshub.com/Rohanpatil4600/YT_comment.mlflow")
dagshub_token=os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] =dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] =dagshub_token
dagshub_url = "https://dagshub.com"
repo_owner= "Rohanpatil4600"
repo_name= "YT_comment"
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# Explicitly check and create experiment
experiment_name = "dvc-pipeline-runs3"
artifact_location = "s3://yt-comment-1/mlruns"

try:
    # Attempt to create the experiment if it doesn't already exist
    mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
except mlflow.exceptions.MlflowException:
    print(f"Experiment '{experiment_name}' already exists. Setting it as active.")
mlflow.set_experiment(experiment_name)

# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        
        logger.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()