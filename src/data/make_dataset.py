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
BASE_URL = os.getenv("BASE_URL")
DATA_ROOT_LOCAL_FOLDER = os.getenv("DATA_ROOT_LOCAL_FOLDER")


class Data:
    """ """

    def __init__(self, input_data: Dict, mode: str = "train"):
        self.input_data = input_data
        self.mode = mode
        self.data_frame = None
        self.data_dict = None
        self.paths = self.get_paths()

    def get_paths(self):
        """Get the paths for different data files."""

        taxi_type = self.input_data['taxi_type']
        year = self.input_data['year']
        month = self.input_data['month']

        # Set the the url of the data file to be downloaded from the NYC taxi server
        file_url = f"{BASE_URL}{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"

        # Set the dict and parquet filenames
        parquet_filename = f"{self.mode}_{taxi_type}_{year}-{month}.parquet"
        dict_filename = f"{self.mode}_{taxi_type}_{year}-{month}.pkl"

        # Set the local file locations
        raw_file_location = os.path.join(DATA_ROOT_LOCAL_FOLDER, "raw/", parquet_filename)
        interim_file_location = os.path.join(DATA_ROOT_LOCAL_FOLDER, "interim/", parquet_filename)
        processed_file_location = os.path.join(DATA_ROOT_LOCAL_FOLDER, "processed/", dict_filename)

        return {"file_url": file_url, "raw": raw_file_location, "interim": interim_file_location, "processed": processed_file_location}

    def download_data(self):
        """
        Download the data from the specified URL.
        """
        self.data_frame = pd.read_parquet(self.paths["file_url"])

        if not os.path.exists(os.path.join(DATA_ROOT_LOCAL_FOLDER, "raw")):
            os.makedirs(os.path.join(DATA_ROOT_LOCAL_FOLDER, "raw"))

        self.data_frame.to_parquet(self.paths["raw"])

        upload_file_to_s3(file_name=self.paths["raw"], bucket=S3_BUCKET, subfolder="raw")

    def prepare_data(self):
        """Prepare the data by performing necessary transformations."""

        self.data_frame["duration"] = self.data_frame.lpep_dropoff_datetime - self.data_frame.lpep_pickup_datetime

        self.data_frame.duration = self.data_frame.duration.dt.total_seconds() / 20

        self.data_frame = self.data_frame[(self.data_frame.duration >= 1) & (self.data_frame.duration <= 60)]

        categorical = ["PULocationID", "DOLocationID"]
        self.data_frame[categorical] = self.data_frame[categorical].astype(str)

        if not os.path.exists(os.path.join(DATA_ROOT_LOCAL_FOLDER, "interim")):
            os.makedirs(os.path.join(DATA_ROOT_LOCAL_FOLDER, "interim"))

        self.data_frame.to_parquet(self.paths["interim"])

        upload_file_to_s3(file_name=self.paths["interim"], bucket=S3_BUCKET, subfolder="interim")

    def prepare_dictionaries(self):
        """Prepare dictionaries for processed data."""

        self.data_frame["PU_DO"] = self.data_frame["PULocationID"] + "_" + self.data_frame["DOLocationID"]
        categorical = ["PU_DO"]
        numerical = ["trip_distance"]
        self.data_dict = self.data_frame[categorical + numerical].to_dict(orient="records")

        if not os.path.exists(os.path.join(DATA_ROOT_LOCAL_FOLDER, "processed")):
            os.makedirs(os.path.join(DATA_ROOT_LOCAL_FOLDER, "processed"))

        with open(self.paths["processed"], "wb") as dict_file:
            pickle.dump(self.data_dict, dict_file)

        upload_file_to_s3(file_name=self.paths["processed"], bucket=S3_BUCKET, subfolder="processed")

    def get_target_values(self):
        """Get the target values from the data frame."""
        return self.data_frame["duration"].values

    def run(self):
        """Run the data processing pipeline."""
        self.download_data()
        self.prepare_data()
        self.prepare_dictionaries()
