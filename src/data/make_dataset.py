# -*- coding: utf-8 -*-
"""_summary_
"""
import os
import pickle
import pandas as pd
from dotenv import load_dotenv
import sys
sys.path.insert(0, './src')


class Data:
    """_summary_"""

    load_dotenv()
    BASE_URL = os.getenv("BASE_URL")

    def __init__(
        self,
        taxi_type: str = "green",
        year: str = "2020",
        month: str = "03",
        mode: str = "train",
        root_folder: str = "data",
    ) -> None:
        self.taxi_type = taxi_type
        self.year = int(year)
        self.month = int(month)
        self.mode = mode
        self.data_frame = None
        self.data_dict = None
        self.root_folder = root_folder
        self.paths = self.get_paths()


    def get_paths(self):
        """_summary_"""
        parquet_filename = (
            f"{self.mode}_{self.taxi_type}_{self.year}-{self.month}.parquet"
        )

        dict_filename = f"{self.mode}_{self.taxi_type}_{self.year}-{self.month}.pkl"

        raw_file_location = os.path.join(self.root_folder, "raw/", parquet_filename)

        interim_file_location = os.path.join(
            self.root_folder, "interim/", parquet_filename
        )
        processed_file_location = os.path.join(
            self.root_folder, "processed/", dict_filename
        )
        file_url = (
            self.BASE_URL
            + f"{self.taxi_type}_tripdata_{self.year:04d}-{self.month:02d}.parquet"
        )

        return {
            "file_url": file_url,
            "raw": raw_file_location,
            "interim": interim_file_location,
            "processed": processed_file_location,
        }


    def download_data(self):
        """
        _summary_
        """
        self.data_frame = pd.read_parquet(self.paths["file_url"])

        if not os.path.exists(os.path.join(self.root_folder, "raw")):
            os.makedirs(os.path.join(self.root_folder, "raw"))

        self.data_frame.to_parquet(self.paths["raw"])

    def prepare_data(self):
        """_summary_"""

        self.data_frame["duration"] = (
            self.data_frame.lpep_dropoff_datetime - self.data_frame.lpep_pickup_datetime
        )

        self.data_frame.duration = self.data_frame.duration.dt.total_seconds() / 60

        self.data_frame = self.data_frame[
            (self.data_frame.duration >= 1) & (self.data_frame.duration <= 60)
        ]

        categorical = [
            "PULocationID",
            "DOLocationID",
        ]
        self.data_frame[categorical] = self.data_frame[categorical].astype(str)

        if not os.path.exists(os.path.join(self.root_folder, "interim")):
            os.makedirs(os.path.join(self.root_folder, "interim"))

        self.data_frame.to_parquet(self.paths["interim"])


    def prepare_dictionaries(self):
        """_summary_"""
        self.data_frame["PU_DO"] = (
            self.data_frame["PULocationID"] + "_" + self.data_frame["DOLocationID"]
        )
        categorical = ["PU_DO"]
        numerical = ["trip_distance"]
        self.data_dict = self.data_frame[categorical + numerical].to_dict(
            orient="records"
        )

        if not os.path.exists(os.path.join(self.root_folder, "processed")):
            os.makedirs(os.path.join(self.root_folder, "processed"))

        with open(self.paths["processed"], "wb") as dict_file:
            pickle.dump(self.data_dict, dict_file)
        

    def get_target_values(self):
        """_summary_"""
        target_column = self.data_frame["duration"].values
        return target_column


    def run(self):
        """_summary_"""
        self.download_data()
        self.prepare_data()
        self.prepare_dictionaries()
