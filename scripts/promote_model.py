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
        # Get the latest version in staging
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
        if not staging_versions:
            pytest.fail("No model version found in Staging")
        latest_staging_version = staging_versions[0].version
        
        # Get current production versions
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        
        # Archive current production models
        for version in prod_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Archived",
                archive_existing_versions=True
            )
            print(f"Archived production model version {version.version}")
        
        # Promote staging model to production
        client.transition_model_version_stage(
            name=model_name,
            version=latest_staging_version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Successfully promoted model version {latest_staging_version} to Production")
        
        # Verify promotion
        new_prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        assert len(new_prod_versions) == 1, "Expected exactly one production model"
        assert new_prod_versions[0].version == latest_staging_version, (
            f"Expected version {latest_staging_version} in production, "
            f"but found version {new_prod_versions[0].version}"
        )
        
    except MlflowException as e:
        pytest.fail(f"MLflow operation failed: {str(e)}")
    except Exception as e:
        pytest.fail(f"Unexpected error during model promotion: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__])