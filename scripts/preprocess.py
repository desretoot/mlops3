import pandas as pd
import numpy as np

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

trainData.to_csv('/home/danil/project/mlops3/datasets/data_preprocess.csv')