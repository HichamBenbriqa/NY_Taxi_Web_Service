"""
Trainer: A class that trains, evaluates and tracks the ML models.

Parameters
    - dict_train (dict): The training data as a dictionary.
    - y_train (array-like): The target variable for training.
    - dict_test (dict): The test data as a dictionary.
    - y_test (array-like): The target variable for testing.
    - params (dict): The parameters for the model.
    - root_folder (str): The root folder to save the model.

Attributes
    - dict_train (dict): The training data as a dictionary.
    - y_train (array-like): The target variable for training.
    - dict_test (dict): The test data as a dictionary.
    - y_test (array-like): The target variable for testing.
    - params (dict): The parameters for the model.
    - pipeline (Pipeline): The trained model pipeline.
    - root_folder (str): The root folder to save the model.
    - pipeline_path (str): The path to save the model pipeline.

"""

import os

import neptune
from dotenv import load_dotenv
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

load_dotenv()
NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT")
NPETUNE_API_TOKEN = os.getenv("NPETUNE_API_TOKEN")
MODEL_ID = os.getenv("MODEL_ID")
S3_BUCKET = os.getenv("S3_BUCKET")


class Trainer:
    """Define Trainer class."""

    def __init__(  # noqa: D417
        self,
        dict_train=None,
        y_train=None,
        dict_test=None,
        y_test=None,
        params=None,
        root_folder="models",
    ):
        """
        Initialize the TrainModel object.

        Parameters
        - dict_train (dict): A dictionary containing the training data.
        - y_train (array-like): The target variable for the training data.
        - dict_test (dict): A dictionary containing the test data.
        - y_test (array-like): The target variable for the test data.
        - params (dict): A dictionary containing the parameters for the model.
        - root_folder (str): The root folder where the model will be saved.
        """
        self.dict_train = dict_train
        self.y_train = y_train
        self.dict_test = dict_test
        self.y_test = y_test
        self.params = params
        self.pipeline = None
        self.root_folder = root_folder
        self.pipeline_path = os.path.join(self.root_folder, "pipeline.joblib")

    def train(self):
        """Train the model using the training data."""
        self.pipeline = make_pipeline(
            DictVectorizer(), RandomForestRegressor(**self.params, n_jobs=-1)
        )
        self.pipeline.fit(self.dict_train, self.y_train)

    def evaluate(self):
        """
        Evaluate the model using the test data.

        Returns
        - rmse (float): The root mean squared error of the model predictions.
        """
        if self.pipeline:
            y_pred = self.pipeline.predict(self.dict_test)
        else:
            self.load_pipeline()
            y_pred = self.pipeline.predict(self.dict_test)
        rmse = mean_squared_error(self.y_test, y_pred, squared=False)
        return rmse

    def predict(self, features):  # noqa: D417
        """
        Make predictions using the trained model.

        Parameters
        - features (dict): The input features for prediction.

        Returns
        - preds (float): The predicted value.
        """
        if self.pipeline:
            preds = self.pipeline.predict(features)
        else:
            self.load_pipeline()
            preds = self.pipeline.predict(features)
        return float(preds[0])

    def save_pipeline(self):
        """Save the trained model pipeline to disk."""
        if not os.path.exists(self.root_folder):
            os.makedirs(self.root_folder)
        dump(self.pipeline, self.pipeline_path)

    def load_pipeline(self):
        """Load the trained model pipeline from disk."""
        self.pipeline = load(self.pipeline_path)

    def upload_to_neptune(self, rmse):  # noqa: D417
        """
        Upload the trained model and related information to Neptune.

        Parameters
        - rmse (float): The root mean squared error of the model predictions.
        """
        run = neptune.init_run(project=NEPTUNE_PROJECT, api_token=NPETUNE_API_TOKEN)
        run["params"] = self.params
        run["rmse"] = rmse
        run["dataset/raw"].track_files(f"s3://{S3_BUCKET}/web-service/raw")
        run["dataset/interim"].track_files(f"s3://{S3_BUCKET}/web-service/interim")
        run["dataset/processed"].track_files(f"s3://{S3_BUCKET}/web-service/processed")

        model_version = neptune.init_model_version(
            model=MODEL_ID, project=NEPTUNE_PROJECT, api_token=NPETUNE_API_TOKEN
        )
        model_version["model"].upload(self.pipeline_path)
        model_version["run/id"] = run["sys/id"].fetch()

        model_version.stop()
        run.stop()
        print("Model uploaded to Neptune")
