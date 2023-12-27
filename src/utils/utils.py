"""Utility functions used throughout the project."""

import logging
import os
from datetime import date

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from hydra import compose, initialize

load_dotenv()
BASE_URL = os.getenv("BASE_URL")
DATA_ROOT_LOCAL_FOLDER = os.getenv("DATA_ROOT_LOCAL_FOLDER", "data")
CONFIG_DIR = os.getenv("CONFIG_DIR")


def get_config(config_path: str = CONFIG_DIR, config_type: str = "data"):
    """
    Read the config file.

    Args:
        config_path (str): The path to the config directory. Defaults to CONFIG_DIR.
        config_type (str): The type of config to read. Defaults to "data".

    Returns:
        tuple or dict: Depending on the config_type, returns a tuple or dictionary
        containing the relevant configuration values.

    Raises:
        ValueError: If an invalid config_type is provided.
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
    Get the previous month given a year and month.

    Args:
        year (int): The year.
        month (int): The month.

    Returns:
        tuple: A tuple containing the year and month of the previous month.
    """
    current_date = date(year, month, 1)
    month, year = (
        (current_date.month - 1, current_date.year)
        if current_date.month != 1
        else (12, current_date.year - 1)
    )

    return year, month


def upload_file_to_s3(file_name, bucket, subfolder):
    """
    Upload a file to an S3 bucket.

    Args:
        file_name (str): The path of the file to be uploaded.
        bucket (str): The name of the S3 bucket.
        subfolder (str): The subfolder within the bucket to upload the file to.

    Returns:
        bool: True if the file was successfully uploaded, False otherwise.
    """
    # If S3 object_name was not specified, use file_name
    key = os.path.join("web-service", subfolder, os.path.basename(file_name))

    s3_client = boto3.client("s3")

    try:
        s3_client.upload_file(file_name, bucket, key)
    except ClientError as e:
        logging.error(e)
        return False
    return True
