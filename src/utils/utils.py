"""_summary_

"""
import os
import logging
import boto3
from botocore.exceptions import ClientError
from datetime import date
from hydra import initialize, compose
from dotenv import load_dotenv

load_dotenv()
BASE_URL = os.getenv("BASE_URL")
DATA_ROOT_LOCAL_FOLDER = os.getenv("DATA_ROOT_LOCAL_FOLDER", "data")
CONFIG_DIR = "../../config"


def get_config(config_path: str = CONFIG_DIR, config_type: str = "data"):
    """
    Read the config file.
    """
    config_file = f"{config_type}.yaml"
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_file)
        if config_type == "data":
            taxi_type = cfg.source.taxi_type
            year = cfg.source.year
            month = cfg.source.month
            return taxi_type, year, month
        if config_type == "model":
            n_estimators = cfg.random_forest_reg.n_estimators
            max_depth = cfg.random_forest_reg.max_depth
            return {"n_estimators": n_estimators, "max_depth": max_depth}
        raise ValueError(f"Invalid config type: {config_type}")


def get_previous_month(year, month):
    """
    Get the previous month and year from the given month and year.
    This method is used to get the test data for the model.
    E.g. if the input month is 1 and input year is 2020, the previous month is 12 and year is 2019.
    The input month and year are the training data. The previous month and year are the test data.
    """
    current_date = date(year, month, 1)
    month, year = (current_date.month - 1, current_date.year) if current_date.month != 1 else (12, current_date.year - 1)

    return year, month


def upload_file_to_s3(file_name, bucket, subfolder):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    key = os.path.join("web-service", subfolder, os.path.basename(file_name))

    # Upload the file
    print(key)
    print(file_name)
    s3_client = boto3.client("s3")
    try:
        response = s3_client.upload_file(file_name, bucket, key)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def get_paths(input_data):
    """Get the paths for different data files."""

    taxi_type = input_data['taxi_type']
    year = input_data['year']
    month = input_data['month']

    # Set the the url of the data file to be downloaded from the NYC taxi server
    file_url = f"{self.BASE_URL}{self.taxi_type}_tripdata_{self.year:04d}-{self.month:02d}.parquet"

    # Set the dict and parquet filenames
    parquet_filename = f"{self.mode}_{taxi_type}_{year}-{month}.parquet"
    dict_filename = f"{self.mode}_{taxi_type}_{year}-{month}.pkl"

    # Set the local file locations
    raw_file_location = os.path.join(DATA_ROOT_LOCAL_FOLDER, "raw/", parquet_filename)
    interim_file_location = os.path.join(DATA_ROOT_LOCAL_FOLDER, "interim/", parquet_filename)
    processed_file_location = os.path.join(DATA_ROOT_LOCAL_FOLDER, "processed/", dict_filename)

    return {"file_url": file_url, "raw": raw_file_location, "interim": interim_file_location, "processed": processed_file_location}
