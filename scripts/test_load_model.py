import mlflow
import pytest
import os
from mlflow.tracking import MlflowClient

# Set the DAGsHub tracking URI
mlflow.set_tracking_uri("https://dagshub.com/Rohanpatil4600/YT_comment.mlflow")

@pytest.mark.parametrize("model_name, version", [
    ("my_model", "7"),  # Replace with your specific model name and version
])
def test_load_model_with_version(model_name, version):
    client = MlflowClient()

    try:
        # Step 1: Get model artifacts path
        model_version = client.get_model_version(model_name, version)
        artifact_uri = model_version.source  # Source path for model artifacts

        print(f"Artifact URI for model '{model_name}' version {version}: {artifact_uri}")

        # Step 2: Download model artifacts locally
        download_path = client.download_artifacts(run_id=model_version.run_id, path="lgbm_model")
        print(f"Model artifacts downloaded to: {download_path}")

        # Step 3: Load model locally
        model = mlflow.pyfunc.load_model(download_path)

        # Validate the model loaded successfully
        assert model is not None, f"Failed to load model '{model_name}' version {version}"
        print(f"Model '{model_name}' version {version} loaded successfully.")

    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")
