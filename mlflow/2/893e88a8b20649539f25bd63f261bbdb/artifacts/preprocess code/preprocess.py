import pandas as pd
import numpy as np
import mlflow
import os

os.environ["MLFLOW_REGISTRY_URI"] = "/home/danil/project/mlops3/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("preprocess")

with mlflow.start_run():

    trainData = pd.read_csv("/home/danil/project/mlops3/datasets/train.csv")

    trainData['Cabin'] = trainData['Cabin'].apply(lambda x: x[0]+x[-1] if pd.notnull(x) and x != '' else x) 

    def fill_na_0(df):
        df.fillna(
            {"HomePlanet": "Earth", 
            "Cabin": "FP",
            "Destination": "TRAPPIST-1e",
            "Age": df.Age.mean().round(),
            "RoomService": 0,
            "FoodCourt": 0,
            "ShoppingMall": 0,
            "Spa": 0,
            "VRDeck": 0,
            "CryoSleep": False, 
            "VIP": False
            }, 
            inplace=True)

    fill_na_0(trainData)

    trainData.drop(["PassengerId", "Name"], axis=1, inplace=True)
    mlflow.log_artifact(local_path="/home/danil/project/mlops3/scripts/preprocess.py",
                        artifact_path="preprocess code")
    mlflow.log_artifact(local_path="/home/danil/project/mlops3/datasets/data_preprocess.csv",
                        artifact_path="train_pre data")

trainData.to_csv('/home/danil/project/mlops3/datasets/data_preprocess.csv')