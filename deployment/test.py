"""Test deployment script."""
import json
import os

import requests
from dotenv import load_dotenv

# from deepdiff import DeepDiff
load_dotenv()
PORT = os.getenv("PORT")
print(PORT)
ride = {"PULocationID": 9, "DOLocationID": 70, "trip_distance": 100}

url = f"http://localhost:{PORT}/predict"

predicted_response = requests.post(url, json=ride).json()
print("predicted response:")
print(json.dumps(predicted_response, indent=2))


"""
expected_response = {
    'duration': 29.3,
    'ride': {'DOLocationID': 70, 'PULocationID': 9, 'trip_distance': 10},
}


diff = DeepDiff(
    expected_response, predicted_response, significant_digits=1
)  # 1 significant digit for float
# so that we dont have to worry if
# the two numbers are off by a few decimals

assert not diff  # if diff dictionnary is empty, then the two dicts are the same

"""
