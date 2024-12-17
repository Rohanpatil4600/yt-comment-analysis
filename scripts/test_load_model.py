import os
import mlflow
import pytest
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

def setup_mlflow_connection():
    
    dagshub_token=os.getenv("DAGSHUB_TOKEN")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")
    
    os.environ["MLFLOW_TRACKING_USERNAME"] =dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] =dagshub_token
    dagshub_url = "https://dagshub.com"
    repo_owner= "Rohanpatil4600"
    repo_name= "YT_comment"
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

    
    # Verify connection
    try:
        client = MlflowClient()
        client.search_registered_models()
        return client
    except Exception as e:
        pytest.fail(f"Failed to connect to MLflow server: {str(e)}")

def get_model_version(client, model_name, stage):
    """Get the specified model version with proper error handling"""
    try:
        model_versions = client.get_latest_versions(model_name, stages=[stage])
        if not model_versions:
            pytest.fail(f"No model versions found for '{model_name}' in '{stage}' stage")
        return model_versions[0]
    except MlflowException as e:
        if e.error_code == "RESOURCE_DOES_NOT_EXIST":
            pytest.fail(f"Model '{model_name}' not found in registry")
        raise

@pytest.mark.parametrize("model_name, stage", [
    ("my_model", "staging"),
])
def test_load_latest_staging_model(model_name, stage):
    """Test loading the latest model version from the specified stage"""
    # Setup MLflow connection
    client = setup_mlflow_connection()
    
    # Get latest model version
    model_version = get_model_version(client, model_name, stage)
    
    try:
        # Construct model URI and load model
        model_uri = f"models:/{model_name}/{model_version.version}"
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Verify model loaded successfully
        assert model is not None, "Model failed to load"
        
        # Additional model verification can be added here
        # For example, checking model signature or testing basic prediction
        
    except Exception as e:
        pytest.fail(f"Failed to load model '{model_name}' version {model_version.version}: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__])