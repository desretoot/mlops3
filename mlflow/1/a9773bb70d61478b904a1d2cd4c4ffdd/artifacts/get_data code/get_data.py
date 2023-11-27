import os
import kaggle
import mlflow
from kaggle.api.kaggle_api_extended import KaggleApi

from mlflow.tracking import MlflowClient

os.environ['KAGGLE_USERNAME'] = "danilsaidyllin"
os.environ['KAGGLE_KEY'] = "3e6a82277e1f607afab69c47ae5634d1"

os.environ["MLFLOW_REGISTRY_URI"] = "/home/danil/project/mlops3/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("get_data")

with mlflow.start_run():

    dataset = 'ajshreim/spaceship-titanic'
    path = '/home/danil/project/mlops3/datasets'

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(dataset, path, unzip=True)
    mlflow.log_artifact(local_path="/home/danil/project/mlops3/scripts/get_data.py",
                        artifact_path="get_data code")
    mlflow.end_run()