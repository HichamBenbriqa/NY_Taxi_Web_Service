"""
Data: a class for handling data download and preparation tasks.

Args:
        input_data (Dict): Input data containing information about taxi type, year,
        and month.
        mode (str, optional): Mode of operation. Defaults to "train".

Attributes:
        input_data (Dict): Input data containing information about taxi type, year,
        and month.
        mode (str): Mode of operation.
        data_frame (pd.DataFrame): Data frame containing the downloaded data.
        data_dict (dict): Processed data in dictionary format.
        paths (dict): Paths for different data files.

Methods:
        get_paths: Get the paths for different data files.
        download_data: Download the data from the specified URL.
        prepare_data: Prepare the data by performing necessary transformations.
        prepare_dictionaries: Prepare dictionaries for processed data.
        get_target_values: Get the target values from the data frame.
        run: Run the data processing pipeline.

"""

import os
import pickle
import sys

sys.path.insert(0, "../")
from dotenv import load_dotenv  # noqa: E402
from utils.utils import upload_file_to_s3  # noqa: E402

load_dotenv()
S3_BUCKET = os.getenv("S3_BUCKET")
BASE_URL = os.getenv("BASE_URL")
DATA_ROOT_LOCAL_FOLDER = os.getenv("DATA_ROOT_LOCAL_FOLDER")


class Data:
    """
    Define the Data class.

    The Data class downloads data from the NYC open API.

    Then it applies transformations to the downloaded data.

    Finally, all the data artifacts are stored in an S3 bucket.
    """

    def __init__(self, input_data: dict, mode: str = "train"):
        """
        Initialize the MakeDataset object.

        Args:
            input_data (Dict): The input data for the dataset.
            mode (str, optional): The mode of the dataset. Defaults to "train".
        """
        self.input_data = input_data
        self.mode = mode
        self.data_frame = None
        self.data_dict = None
        self.paths = self.get_paths()

    def get_paths(self):
        """
        Get the paths for different data files.

        Returns
            dict: A dictionary containing the file URLs and local file locations.
                - "file_url" (str): The URL of the data file to be downloaded.
                - "raw" (str): The local file location for the raw data file.
                - "interim" (str): The local file location for the interim data file.
                - "processed" (str): The local file location for the processed files.
        """
        taxi_type = self.input_data["taxi_type"]
        year = self.input_data["year"]
        month = self.input_data["month"]

        # Set the the url of the data file to be downloaded from the NYC taxi server
        file_url = f"{BASE_URL}{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"

        # Set the dict and parquet filenames
        parquet_filename = f"{self.mode}_{taxi_type}_{year}-{month}.parquet"
        dict_filename = f"{self.mode}_{taxi_type}_{year}-{month}.pkl"

        # Set the local file locations
        raw_file_location = os.path.join(
            DATA_ROOT_LOCAL_FOLDER, "raw/", parquet_filename
        )
        interim_file_location = os.path.join(
            DATA_ROOT_LOCAL_FOLDER, "interim/", parquet_filename
        )
        processed_file_location = os.path.join(
            DATA_ROOT_LOCAL_FOLDER, "processed/", dict_filename
        )

        return {
            "file_url": file_url,
            "raw": raw_file_location,
            "interim": interim_file_location,
            "processed": processed_file_location,
        }

    def download_data(self, upload_s3=True):
        """
        Download the data from the specified URL.

        This method downloads the data from the specified URL and saves it locally in
        the "raw" folder.
        It also uploads the raw data file to an S3 bucket.
        """
        if not os.path.exists(os.path.join(DATA_ROOT_LOCAL_FOLDER, "raw")):
            os.makedirs(os.path.join(DATA_ROOT_LOCAL_FOLDER, "raw"))

        self.data_frame.to_parquet(self.paths["raw"])

        if upload_s3:
            upload_file_to_s3(
                file_name=self.paths["raw"], bucket=S3_BUCKET, subfolder="raw"
            )

    def prepare_data(self, upload_s3=True):
        """
        Prepare the data by performing necessary transformations.

        This method performs the following transformations on the data:
        1. Calculates the duration of each trip in seconds.
        2. Filters out trips with duration less than 1 second or greater than 60.
        3. Converts the categorical columns 'PULocationID' and 'DOLocationID' to string.
        4. Creates the 'interim' folder if it doesn't exist in the data root folder.
        5. Saves the transformed data frame as a parquet file in the 'interim' folder.
        6. Uploads the parquet file to the specified S3 bucket and subfolder.
        """
        self.data_frame["duration"] = (
            self.data_frame.lpep_dropoff_datetime - self.data_frame.lpep_pickup_datetime
        )

        self.data_frame.duration = self.data_frame.duration.dt.total_seconds() / 20

        self.data_frame = self.data_frame[
            (self.data_frame.duration >= 1) & (self.data_frame.duration <= 60)
        ]

        categorical = ["PULocationID", "DOLocationID"]
        self.data_frame[categorical] = self.data_frame[categorical].astype(str)

        if not os.path.exists(os.path.join(DATA_ROOT_LOCAL_FOLDER, "interim")):
            os.makedirs(os.path.join(DATA_ROOT_LOCAL_FOLDER, "interim"))

        self.data_frame.to_parquet(self.paths["interim"])

        if upload_s3:
            upload_file_to_s3(
                file_name=self.paths["interim"], bucket=S3_BUCKET, subfolder="interim"
            )

    def prepare_dictionaries(self, upload_s3=True):
        """
        Prepare dictionaries for processed data.

        This method prepares dictionaries for the processed data:
        1. It creates a new column in the data frame called "PU_DO" by concatenating the
        "PULocationID" and "DOLocationID" columns.
        2. It then selects the categorical and numerical columns from the data frame and
        converts them into a dictionary format.
        3. The resulting dictionary is saved as a pickle file and uploaded to an S3.
        """
        self.data_frame["PU_DO"] = (
            self.data_frame["PULocationID"] + "_" + self.data_frame["DOLocationID"]
        )
        categorical = ["PU_DO"]
        numerical = ["trip_distance"]
        self.data_dict = self.data_frame[categorical + numerical].to_dict(
            orient="records"
        )

        if not os.path.exists(os.path.join(DATA_ROOT_LOCAL_FOLDER, "processed")):
            os.makedirs(os.path.join(DATA_ROOT_LOCAL_FOLDER, "processed"))

        with open(self.paths["processed"], "wb") as dict_file:
            pickle.dump(self.data_dict, dict_file)

        if upload_s3:
            upload_file_to_s3(
                file_name=self.paths["processed"],
                bucket=S3_BUCKET,
                subfolder="processed",
            )

    def get_target_values(self):
        """
        Get the target values from the data frame.

        Returns
            numpy.ndarray: An array containing the target values.
        """
        return self.data_frame["duration"].values

    def run(self):
        """
        Run the data processing pipeline.

        This method executes the necessary steps to process the data,
        including downloading the data, preparing it, and preparing the dictionaries.
        """
        self.download_data()
        self.prepare_data()
        self.prepare_dictionaries()
