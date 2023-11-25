import os
import kaggle

os.environ['KAGGLE_USERNAME'] = "danilsaidyllin"
os.environ['KAGGLE_KEY'] = "3e6a82277e1f607afab69c47ae5634d1"

from kaggle.api.kaggle_api_extended import KaggleApi

dataset = 'ajshreim/spaceship-titanic'
path = '/home/danil/project/mlops3/datasets'

api = KaggleApi()
api.authenticate()

api.dataset_download_files(dataset, path, unzip=True)
