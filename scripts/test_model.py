from catboost import CatBoostClassifier
import pickle
import pandas as pd

df = pd.read_csv('/home/danil/project/mlops3/datasets/data_test.csv')

x = df.drop('Transported', axis = 1)
y = df['Transported']
model = CatBoostClassifier()

with open('/home/danil/project/mlops3/models/data.pickle', 'rb') as f:
    model = pickle.load(f)

score = model.predict(x)
print("score=", score)
