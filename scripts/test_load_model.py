import mlflow.pyfunc
import pytest
from mlflow.tracking import MlflowClient

# Set the tracking URI to DAGsHub
mlflow.set_tracking_uri("https://dagshub.com/Rohanpatil4600/YT_comment.mlflow")

@pytest.mark.parametrize("model_name, stage", [
    ("my_model", "staging"),])
def test_load_latest_staging_model(model_name, stage):
    client = MlflowClient()
    
    # Fetch all registered model versions for the specified model
    all_versions = client.search_model_versions(f"name='{model_name}'")
    staging_versions = [v for v in all_versions if v.current_stage == stage]
    
    # Get the latest version in the 'staging' stage
    if staging_versions:
        latest_version = sorted(staging_versions, key=lambda v: int(v.version), reverse=True)[0]
    else:
        pytest.fail(f"No model found in the '{stage}' stage for '{model_name}'")

    try:
        # Load the model using its version
        model_uri = f"models:/{model_name}/{latest_version.version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Ensure the model loads successfully
        assert model is not None, "Model failed to load"
        print(f"Model '{model_name}' version {latest_version.version} loaded successfully from '{stage}' stage.")

    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")
