"""Unit tests of the Data class methods."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append("src/data")
from dotenv import load_dotenv  # noqa: E402
from make_dataset import Data  # noqa: E402

load_dotenv()
DATA_ROOT_LOCAL_FOLDER = os.getenv("DATA_ROOT_LOCAL_FOLDER")


@pytest.fixture
def input_data():
    """
    Return a dictionary containing the input data for the test.

    Return:
        dict: A dictionary with keys 'taxi_type', 'year', and 'month'.
              'taxi_type' (str): The type of taxi.
              'year' (int): The year.
              'month' (int): The month.
    """
    return {"taxi_type": "green", "year": 2022, "month": 1}


@pytest.fixture
def raw_data_frame():
    """
    Create a sample DataFrame with dummy data.

    Returns
        pandas.DataFrame: A DataFrame containing dummy data.
    """
    return pd.DataFrame(
        {
            "lpep_pickup_datetime": pd.to_datetime(
                ["2022-01-01 00:00:00", "2022-01-01 00:01:00"]
            ),
            "lpep_dropoff_datetime": pd.to_datetime(
                ["2022-01-01 00:01:00", "2022-01-01 00:02:00"]
            ),
            "PULocationID": [1, 2],
            "DOLocationID": [3, 4],
            "trip_distance": [1.5, 2.0],
        }
    )


@pytest.fixture
def sample_interim_dataframe():
    """
    Generate a sample interim DataFrame.

    Returns
        pandas.DataFrame: A DataFrame containing sample interim data.
    """
    return pd.DataFrame(
        {
            "PULocationID": ["1", "2", "3"],
            "DOLocationID": ["4", "5", "6"],
            "trip_distance": [1.5, 2.0, 3.0],
            "duration": [1.0, 2.0, 3.0],
        }
    )


def test_get_paths(input_data):
    """
    Test case for the get_paths method of the Data class.

    Args:
        input_data: The input data for the test.

    Returns:
        None
    """
    data = Data(input_data)
    paths = data.get_paths()
    assert isinstance(paths, dict)
    assert "file_url" in paths
    assert "raw" in paths
    assert "interim" in paths
    assert "processed" in paths


def test_download_data(input_data):
    """
    Test function to verify the download_data method of the Data class.

    Args:
        input_data: The input data for the test.
        data_frame: The data frame to be used for testing.

    Returns:
        None
    """
    data = Data(input_data)
    data.data_frame = pd.DataFrame()
    data.download_data(upload_s3=False)

    # Check if the data is downloaded and uploaded correctly
    assert os.path.exists(data.paths["raw"])

    # Delete the downloaded file
    os.remove(data.paths["raw"])
    assert not os.path.exists(data.paths["raw"])


def test_prepare_data_transformations(input_data, raw_data_frame):
    """
    Verify if the transformations in the prepare_data method are applied correctly.

    Args:
        input_data: The input data for preparing.
        raw_data_frame: The data frame to be used.

    Returns:
        None
    """
    data = Data(input_data)
    data.data_frame = raw_data_frame
    data.prepare_data(upload_s3=False)

    # Check if the duration is calculated correctly
    assert "duration" in data.data_frame.columns
    assert data.data_frame.duration.dtype == np.float64

    # Check if the duration is calculated in seconds
    assert all(
        data.data_frame.duration
        == (
            data.data_frame.lpep_dropoff_datetime - data.data_frame.lpep_pickup_datetime
        ).dt.total_seconds()
        / 20
    )

    # Check if the trips with duration less than 1 second or greater than 60 seconds
    # are filtered out
    assert all((data.data_frame.duration >= 1) & (data.data_frame.duration <= 60))

    # Check if the categorical columns are converted to string type
    assert data.data_frame["PULocationID"].dtype == object
    assert data.data_frame["DOLocationID"].dtype == object

    # Check if the 'interim' folder is created
    assert os.path.exists(os.path.join(DATA_ROOT_LOCAL_FOLDER, "interim"))

    # Check if the parquet file is saved in the 'interim' folder
    assert os.path.exists(data.paths["interim"])


def test_prepare_dictionaries(input_data, sample_interim_dataframe):
    """
    Test function for preparing dictionaries in the Data class.

    Args:
        input_data (list): The input data for the Data class.
        sample_interim_dataframe (DataFrame): The data frame to be used in the test.
    """
    data = Data(input_data)
    data.data_frame = sample_interim_dataframe
    data.prepare_dictionaries(upload_s3=False)

    # Check if the "PU_DO" column is added to the data frame
    assert "PU_DO" in data.data_frame.columns

    # Check if the "PU_DO" column is created correctly by concatenating "PULocationID"
    # and "DOLocationID"
    expected_pu_do = (
        sample_interim_dataframe["PULocationID"].astype(str)
        + "_"
        + sample_interim_dataframe["DOLocationID"].astype(str)
    )
    assert all(data.data_frame["PU_DO"] == expected_pu_do)

    # Check if the dictionary is created correctly
    expected_dict = sample_interim_dataframe[["PU_DO", "trip_distance"]].to_dict(
        orient="records"
    )
    assert data.data_dict == expected_dict

    # Check if the processed folder is created
    assert os.path.exists(os.path.join(DATA_ROOT_LOCAL_FOLDER, "processed"))

    # Check if the dictionary file is saved
    assert os.path.exists(data.paths["processed"])

    # Delete the interim file
    os.remove(data.paths["interim"])
    assert not os.path.exists(data.paths["interim"])

    # Delete the interim file
    os.remove(data.paths["processed"])
    assert not os.path.exists(data.paths["processed"])


def test_get_target_values(input_data, sample_interim_dataframe):
    """
    Test case for the get_target_values method of the Data class.

    Args:
        input_data: The input data for the test.
        sample_interim_dataframe: The data frame to be used for testing.

    Returns:
        None
    """
    data = Data(input_data)
    data.data_frame = sample_interim_dataframe
    target_values = data.get_target_values()
    assert isinstance(target_values, np.ndarray)
