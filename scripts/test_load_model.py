import mlflow.pyfunc
import pytest
from mlflow.tracking import MlflowClient

# Set the tracking URI to DAGsHub
mlflow.set_tracking_uri("https://dagshub.com/Rohanpatil4600/YT_comment.mlflow")

@pytest.mark.parametrize("model_name, stage", [
    ("my_model", "staging"),])
def test_load_latest_staging_model(model_name, stage):
    client = MlflowClient()
    
    try:
        # Fetch all model versions with search_model_versions
        versions = client.search_model_versions(f"name='{model_name}'")

        # Filter versions by stage
        staging_versions = [
            v for v in versions if v.current_stage == stage
        ]

        # Ensure at least one version exists in the specified stage
        assert staging_versions, f"No model found in the '{stage}' stage for '{model_name}'"

        # Get the latest version (numerically highest)
        latest_version = max(staging_versions, key=lambda v: int(v.version))

        # Load the model from the Model Registry
        model_uri = f"models:/{model_name}/{latest_version.version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Validate the model loaded successfully
        assert model is not None, "Model failed to load"
        print(f"Model '{model_name}' version {latest_version.version} loaded successfully from '{stage}' stage.")

    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")
