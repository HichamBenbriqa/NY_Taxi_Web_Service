"""
    summary
"""
import os
from joblib import dump, load
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer


class Trainer:
    """
    summary
    """

    def __init__(
        self,
        dict_train=None,
        y_train=None,
        dict_test=None,
        y_test=None,
        params=None,
        root_folder=("models"),
    ):
        self.dict_train = dict_train
        self.y_train = y_train
        self.dict_test = dict_test
        self.y_test = y_test
        self.params = params
        self.pipeline = None
        self.root_folder = root_folder
        self.pipeline_path = os.path.join(self.root_folder, "pipeline.joblib")

    def train(self):
        """_summary_"""
        self.pipeline = make_pipeline(
            DictVectorizer(), RandomForestRegressor(**self.params, n_jobs=-1)
        )
        self.pipeline.fit(self.dict_train, self.y_train)

    def evaluate(self):
        """_summary_"""
        if self.pipeline:
            y_pred = self.pipeline.predict(self.dict_test)
        else:
            self.load_pipeline()
            y_pred = self.pipeline.predict(self.dict_test)
        rmse = mean_squared_error(self.y_test, y_pred, squared=False)
        return rmse

    def predict(self, features):
        """_summary_"""
        if self.pipeline:
            preds = self.pipeline.predict(features)
        else:
            self.load_pipeline()
            preds = self.pipeline.predict(features)
        return float(preds[0])

    def save_pipeline(self):
        """_summary_"""
        dump(self.pipeline, self.pipeline_path)

    def load_pipeline(self):
        """_summary_"""
        self.pipeline = load(self.pipeline_path)
