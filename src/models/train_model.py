"""
    summary
"""
import os
import neptune
from joblib import dump, load
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer

load_dotenv()
NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT")
NPETUNE_API_TOKEN = os.getenv("NPETUNE_API_TOKEN")
MODEL_ID = os.getenv("MODEL_ID")
print(NEPTUNE_PROJECT, NPETUNE_API_TOKEN, MODEL_ID)

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
        root_folder="models",
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
        if not os.path.exists(self.root_folder):
            os.makedirs(self.root_folder)
        dump(self.pipeline, self.pipeline_path)

    def load_pipeline(self):
        """_summary_"""
        self.pipeline = load(self.pipeline_path)
    
    def upload_to_neptune(self):
        model_version = neptune.init_model_version(model=MODEL_ID, project=NEPTUNE_PROJECT, api_token=NPETUNE_API_TOKEN,)
        model_version["model"].upload(self.pipeline_path)
        #model_version["validation/dataset"].track_files(self.dict_test)
        #model_version["validation/acc"] = 0.97
        model_version.stop()
        print("Model uploaded to Neptune")