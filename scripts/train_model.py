from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
import pickle
import os
import mlflow

# os.environ["MLFLOW_REGISTRY_URI"] = "/home/danil/project/mlops3/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("data_train")

with mlflow.start_run():

    trainData = pd.read_csv('/home/danil/project/mlops3/datasets/data_train.csv')

    X = trainData.drop(["Transported"], axis=1)
    y = trainData.Transported

    categorical_features_indices = np.where(X.dtypes == 'object')[0]

    model = CatBoostClassifier(iterations=1500,
                            loss_function='Logloss',  
                            eval_metric='Accuracy',  
                            logging_level='Silent',
                            early_stopping_rounds=300,
                            cat_features=categorical_features_indices
                            )

    grid = {
        'depth': [4, 6, 8],
        'learning_rate': [0.04, 0.06, 0.08]
    }

    grid_search_result = model.grid_search(grid, 
                                        X=X, 
                                        y=y,
                                        cv=5,
                                        train_size=0.8,
                                        plot=True)
    print("Best model parameters: " + str(grid_search_result['params'])) 

    with open('/home/danil/project/mlops3/models/data.pickle', 'wb') as f:
        pickle.dump(model, f)

    mlflow.log_artifact(local_path="/home/danil/project/mlops3/scripts/train_model.py",
                        artifact_path="data_train code")
    mlflow.sklearn.log_model(model, "model")
    mlflow.end_run()