import pandas as pd
import numpy as np
import os
import mlflow

# os.environ["MLFLOW_REGISTRY_URI"] = "/home/danil/project/mlops3/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("train_test_split")

with mlflow.start_run():
    df = pd.read_csv('/home/danil/project/mlops3/datasets/data_preprocess.csv')

    idxs = np.array(df.index.values)
    np.random.shuffle(idxs)
    l = int(len(df)*0.7)
    train_idxs = idxs[:l]
    test_idxs = idxs[l+1:]

    mlflow.log_artifact(local_path="/home/danil/project/mlops3/scripts/train_test_split.py",
                        artifact_path="train_test_split code")
    # mlflow.log_artifact(local_path="/home/danil/project/mlops3/datasets/data_train.csv",
    #                     artifact_path="train data")
    # mlflow.log_artifact(local_path="/home/danil/project/mlops3/datasets/data_test.csv",
    #                     artifact_path="test data")
    mlflow.end_run()

df.loc[train_idxs, :].to_csv('/home/danil/project/mlops3/datasets/data_train.csv', index=False)
df.loc[test_idxs, :].to_csv('/home/danil/project/mlops3/datasets/data_test.csv', index=False)
