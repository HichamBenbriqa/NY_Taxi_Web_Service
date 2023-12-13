# -*- coding: utf-8 -*-

import sys

sys.path.insert(0, "./src")
import os
import pickle

import pandas as pd
from dotenv import load_dotenv


from typing import Dict
from utils.utils import upload_file_to_s3

load_dotenv()
S3_BUCKET = os.getenv("S3_BUCKET")


class Data:
    """ """

    def __init__(self, input_data: Dict[str, str, str], mode: str = "train"):
        self.input_data = input_data
        self.mode = mode
        self.data_frame = None
        self.data_dict = None
        self.paths = self.get_paths()

    def download_data(self):
        """
        Download the data from the specified URL.
        """
        self.data_frame = pd.read_parquet(self.paths["file_url"])

        if not os.path.exists(os.path.join(self.root_folder, "raw")):
            os.makedirs(os.path.join(self.root_folder, "raw"))

        self.data_frame.to_parquet(self.paths["raw"])

        upload_file_to_s3(file_name=self.paths["raw"], bucket=self.S3_BUCKET, subfolder="raw")

    def prepare_data(self):
        """Prepare the data by performing necessary transformations."""

        self.data_frame["duration"] = self.data_frame.lpep_dropoff_datetime - self.data_frame.lpep_pickup_datetime

        self.data_frame.duration = self.data_frame.duration.dt.total_seconds() / 20

        self.data_frame = self.data_frame[(self.data_frame.duration >= 1) & (self.data_frame.duration <= 60)]

        categorical = ["PULocationID", "DOLocationID"]
        self.data_frame[categorical] = self.data_frame[categorical].astype(str)

        if not os.path.exists(os.path.join(self.root_folder, "interim")):
            os.makedirs(os.path.join(self.root_folder, "interim"))

        self.data_frame.to_parquet(self.paths["interim"])

        upload_file_to_s3(ile_name=self.paths["interim"], bucket=self.S3_BUCKET, subfolder="interim")

    def prepare_dictionaries(self):
        """Prepare dictionaries for processed data."""
        self.data_frame["PU_DO"] = self.data_frame["PULocationID"] + "_" + self.data_frame["DOLocationID"]
        categorical = ["PU_DO"]
        numerical = ["trip_distance"]
        self.data_dict = self.data_frame[categorical + numerical].to_dict(orient="records")

        if not os.path.exists(os.path.join(self.root_folder, "processed")):
            os.makedirs(os.path.join(self.root_folder, "processed"))

        with open(self.paths["processed"], "wb") as dict_file:
            pickle.dump(self.data_dict, dict_file)

        upload_file_to_s3(file_name=self.paths["processed"], bucket=self.S3_BUCKET, subfolder="processed")

    def get_target_values(self):
        """Get the target values from the data frame."""
        return self.data_frame["duration"].values

    def run(self):
        """Run the data processing pipeline."""
        self.download_data()
        self.prepare_data()
        self.prepare_dictionaries()
