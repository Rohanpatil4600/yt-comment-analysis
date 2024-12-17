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
        # Get the registered model details
        registered_model = client.get_registered_model(model_name)

        # Extract all model versions
        all_versions = registered_model.latest_versions

        # Filter versions by stage
        staging_versions = [
            version for version in all_versions if version.current_stage == stage
        ]

        # Ensure we have at least one version in staging
        assert staging_versions, f"No model found in the '{stage}' stage for '{model_name}'"

        # Get the latest version (highest version number)
        latest_version = max(staging_versions, key=lambda v: int(v.version))

        # Load the model using the latest version
        model_uri = f"models:/{model_name}/{latest_version.version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Ensure the model loads successfully
        assert model is not None, "Model failed to load"
        print(f"Model '{model_name}' version {latest_version.version} loaded successfully from '{stage}' stage.")

    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")
