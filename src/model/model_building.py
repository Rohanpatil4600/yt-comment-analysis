import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.feature_extraction.text import TfidfVectorizer

# Logging configuration
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    """Apply TF-IDF with ngrams to the data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values

        # Perform TF-IDF transformation
        X_train_tfidf = vectorizer.fit_transform(X_train)

        logger.debug(f"TF-IDF transformation complete. Train shape: {X_train_tfidf.shape}")
        return X_train_tfidf, y_train, vectorizer
    except Exception as e:
        logger.error('Error during TF-IDF transformation: %s', e)
        raise


def train_lgbm(X_train: np.ndarray, y_train: np.ndarray, learning_rate: float, max_depth: int, n_estimators: int) -> lgb.LGBMClassifier:
    """Train a LightGBM model."""
    try:
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            metric="multi_logloss",
            is_unbalance=True,
            class_weight="balanced",
            reg_alpha=0.1,
            reg_lambda=0.1,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators
        )
        model.fit(X_train, y_train)
        logger.debug('LightGBM model training completed')
        return model
    except Exception as e:
        logger.error('Error during LightGBM model training: %s', e)
        raise


def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def initialize_mlflow_experiment(experiment_name, artifact_location):
    """Create an MLflow experiment if it doesn't exist."""
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            # Create experiment if it doesn't exist
            mlflow.create_experiment(name=experiment_name, artifact_location=artifact_location)
            logger.debug(f"Experiment '{experiment_name}' created at {artifact_location}")
        else:
            logger.debug(f"Experiment '{experiment_name}' already exists")
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.error(f"Failed to initialize MLflow experiment: {e}")
        raise


def main():
    try:
        # Get root directory and resolve the path for params.yaml
        root_dir = get_root_directory()

        # Load parameters
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        max_features = params['model_building']['max_features']
        ngram_range = tuple(params['model_building']['ngram_range'])
        learning_rate = params['model_building']['learning_rate']
        max_depth = params['model_building']['max_depth']
        n_estimators = params['model_building']['n_estimators']

        # Initialize MLflow Experiment
        experiment_name = "model_building_experiment"
        artifact_location = "s3://yt-comment-1/mlruns"  # Update this to your artifact storage path
        initialize_mlflow_experiment(experiment_name, artifact_location)

        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_param("max_features", max_features)
            mlflow.log_param("ngram_range", ngram_range)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("n_estimators", n_estimators)

            # Load training data
            train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))

            # Apply TF-IDF
            X_train_tfidf, y_train, vectorizer = apply_tfidf(train_data, max_features, ngram_range)

            # Train LightGBM model
            model = train_lgbm(X_train_tfidf, y_train, learning_rate, max_depth, n_estimators)

            # Save and log vectorizer
            vectorizer_path = os.path.join(root_dir, 'tfidf_vectorizer.pkl')
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            mlflow.log_artifact(vectorizer_path)

            # Save and log model
            model_path = os.path.join(root_dir, 'lgbm_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            mlflow.lightgbm.log_model(model, "lgbm_model")

            logger.debug('Model and vectorizer logged to MLflow')

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
