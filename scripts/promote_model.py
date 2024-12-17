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
    
    repo_owner = "Rohanpatil4600"
    repo_name = "YT_comment"
    tracking_uri = f"https://{repo_owner}:{dagshub_token}@dagshub.com/{repo_owner}/{repo_name}.mlflow"
    
    mlflow.set_tracking_uri(tracking_uri)
    os.environ["MLFLOW_TRACKING_TOKEN"] = dagshub_token
    
    try:
        client = MlflowClient()
        client.search_registered_models()
        return client
    except Exception as e:
        pytest.fail(f"Failed to connect to MLflow server: {str(e)}")

def test_promote_model_to_production():
    """Test promoting the latest model to production and second-latest to archive"""
    client = setup_mlflow_connection()
    model_name = "my_model"
    
    try:
        # Get all model versions sorted by version number (descending)
        all_versions = client.search_model_versions(f"name='{model_name}'")
        sorted_versions = sorted(all_versions, key=lambda x: int(x.version), reverse=True)
        
        if len(sorted_versions) < 2:
            pytest.fail("Need at least 2 model versions to perform promotion")
        
        latest_version = sorted_versions[0].version
        second_latest_version = sorted_versions[1].version
        
        print(f"Latest version: {latest_version}")
        print(f"Second latest version: {second_latest_version}")
        
        # First, transition the second-latest version to Archive
        print(f"Moving version {second_latest_version} to Archive")
        client.transition_model_version_stage(
            name=model_name,
            version=second_latest_version,
            stage="Archived"
        )
        
        # Then, promote the latest version to Production
        print(f"Moving version {latest_version} to Production")
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Production"
        )
        
        # Verify the changes
        updated_latest = client.get_model_version(model_name, latest_version)
        updated_second_latest = client.get_model_version(model_name, second_latest_version)
        
        assert updated_latest.current_stage == "Production", \
            f"Expected version {latest_version} to be in Production, but it's in {updated_latest.current_stage}"
        
        assert updated_second_latest.current_stage == "Archived", \
            f"Expected version {second_latest_version} to be in Archived, but it's in {updated_second_latest.current_stage}"
        
        print("\nModel promotion completed successfully:")
        print(f"- Version {latest_version} is now in Production")
        print(f"- Version {second_latest_version} is now in Archived")
        
    except MlflowException as e:
        pytest.fail(f"MLflow operation failed: {str(e)}")
    except Exception as e:
        pytest.fail(f"Unexpected error during model promotion: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__])