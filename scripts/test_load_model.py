import mlflow.pyfunc
import pytest

# Set the tracking URI to DAGsHub
mlflow.set_tracking_uri("https://dagshub.com/Rohanpatil4600/YT_comment.mlflow")

@pytest.mark.parametrize("model_name, version", [
    ("my_model", "7"),  # Replace with the latest version explicitly
])
def test_load_model_with_version(model_name, version):
    try:
        # Define the model URI directly
        model_uri = f"models:/{model_name}/{version}"
        
        # Load the model
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Ensure the model loaded successfully
        assert model is not None, f"Failed to load model '{model_name}' version {version}"
        print(f"Model '{model_name}' version {version} loaded successfully.")

    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")
