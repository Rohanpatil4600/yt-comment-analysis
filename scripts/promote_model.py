import os
import pytest
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

def setup_mlflow_connection():
    """Configure MLflow connection with proper authentication"""
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if not dagshub_token:
        pytest.fail("DAGSHUB_TOKEN environment variable is not set")
    
    # Set up MLflow tracking URI with authentication
    repo_owner = "Rohanpatil4600"
    repo_name = "YT_comment"
    tracking_uri = f"https://{repo_owner}:{dagshub_token}@dagshub.com/{repo_owner}/{repo_name}.mlflow"
    
    mlflow.set_tracking_uri(tracking_uri)
    os.environ["MLFLOW_TRACKING_TOKEN"] = dagshub_token
    
    try:
        client = MlflowClient()
        # Verify connection
        client.search_registered_models()
        return client
    except Exception as e:
        pytest.fail(f"Failed to connect to MLflow server: {str(e)}")

def test_promote_model_to_production():
    """Test promoting the latest staging model to production"""
    # Setup connection
    client = setup_mlflow_connection()
    model_name = "my_model"
    
    try:
        # First, get the current production model(s) and archive them
        current_prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        for version in current_prod_versions:
            print(f"Archiving current production model version {version.version}")
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Archived"
            )
        
        # Now get the latest staging model
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
        if not staging_versions:
            pytest.fail("No model version found in Staging")
        
        staging_version = staging_versions[0].version
        print(f"Found staging model version {staging_version}")
        
        # Promote staging model to production
        print(f"Promoting model version {staging_version} to Production")
        client.transition_model_version_stage(
            name=model_name,
            version=staging_version,
            stage="Production"
        )
        
        # Verify the changes
        new_prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        archived_versions = client.get_latest_versions(model_name, stages=["Archived"])
        
        # Verify production model
        assert len(new_prod_versions) == 1, "Expected exactly one production model"
        assert new_prod_versions[0].version == staging_version, (
            f"Expected version {staging_version} in production, "
            f"but found version {new_prod_versions[0].version}"
        )
        
        # Verify archived models
        assert len(archived_versions) == len(current_prod_versions), (
            "Number of archived models doesn't match previous production models"
        )
        
        print("Model promotion completed successfully:")
        print(f"- New production model: version {staging_version}")
        print(f"- Archived models: {[v.version for v in archived_versions]}")
        
    except MlflowException as e:
        pytest.fail(f"MLflow operation failed: {str(e)}")
    except Exception as e:
        pytest.fail(f"Unexpected error during model promotion: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__])