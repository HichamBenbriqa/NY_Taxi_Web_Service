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
CONFIG_DIR = os.getenv("CONFIG_DIR")
AWS_REGION = os.getenv("AWS_REGION")


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
    # print(key)
    # print(file_name)

    # print(os.environ)

    # # Access AWS_ACCESS_KEY_ID environment variable
    # aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    # AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
    # ECR_REGISTRY = os.environ.get("ECR_REGISTRY")

    # if aws_access_key_id or AWS_SECRET_ACCESS_KEY or ECR_REGISTRY:
    #     print(f"AWS Access Key ID: {aws_access_key_id}")
    #     print(f"AWS_SECRET_ACCESS_KEY: {AWS_SECRET_ACCESS_KEY}")
    #     print(f"AWS_SECRET_ACCESS_KEY: {ECR_REGISTRY}")
    # else:
    #     print("AWS_ACCESS_KEY_ID is not set.")

    # Initialize a Boto3 SSM client
    ssm_client = boto3.client('ssm', region_name=AWS_REGION)

    # Specify the name of the Parameter Store parameter where your access keys are stored
    parameter_aws_access_key_id = 'access_key_id'
    parameter_aws_secret_access_key = 'secret_access_key'

    # Retrieve the parameter value (access keys) from Parameter Store
    response_aws_access_key_id = ssm_client.get_parameter(Name=parameter_aws_access_key_id, WithDecryption=True)
    response_aws_secret_access_key = ssm_client.get_parameter(Name=parameter_aws_secret_access_key, WithDecryption=True)

    # Parse the response to extract the access keys
    # parameter_value = response['Parameter']['Value'].splitlines()
    aws_access_key_id, aws_secret_access_key = response_aws_access_key_id['Parameter']['Value'], response_aws_secret_access_key['Parameter']['Value']

    # Use the retrieved access keys for AWS SDK or API calls
    # For example, you can set them as environment variables for AWS SDK clients
    os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
    os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key

    s3_client = boto3.client("s3")
    # session = boto3.Session(aws_access_key_id=aws_access_key_id, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

    # s3_client = session.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, key)
    except ClientError as e:
        logging.error(e)
        return False
    return True
