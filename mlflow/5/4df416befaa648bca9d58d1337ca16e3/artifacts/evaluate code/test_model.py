from catboost import CatBoostClassifier
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
import os
import mlflow

# os.environ["MLFLOW_REGISTRY_URI"] = "/home/danil/project/mlops3/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("test_model")

with mlflow.start_run():

    df = pd.read_csv('/home/danil/project/mlops3/datasets/data_test.csv')

    x = df.drop('Transported', axis = 1)
    y = df['Transported']
    categorical_features_indices = np.where(x.dtypes == 'object')[0]
    model = CatBoostClassifier()

    with open('/home/danil/project/mlops3/models/data.pickle', 'rb') as f:
        model = pickle.load(f)

    y_pred_validation= model.predict(x)

    score = accuracy_score(y_pred_validation, y.astype(str))
    mlflow.log_artifact(local_path="/home/danil/project/mlops3/scripts/test_model.py",
                        artifact_path="evaluate code")
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_metric("test_score", score)
    mlflow.end_run()