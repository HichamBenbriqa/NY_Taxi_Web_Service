"""_summary_."""
import json
import os
import subprocess
import time
from time import gmtime, strftime

import boto3
import neptune
import sagemaker
from dotenv import load_dotenv

load_dotenv()
NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT")
NPETUNE_API_TOKEN = os.getenv("NPETUNE_API_TOKEN")
MODEL_ID = os.getenv("MODEL_ID")
AWS_SAGEMAKER_ROLE = os.getenv("AWS_SAGEMAKER_ROLE")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET = os.getenv("S3_BUCKET")


class Deployer:
    """summary."""

    def __init__(self, sagemaker_client, model_artifacts_tar, boto_session):
        """summary."""
        self.sagemaker_client = sagemaker_client
        self.model_artifacts_tar = model_artifacts_tar
        self.boto_session = boto_session

    def get_production_ready_model(self):
        """_summary_."""
        model = neptune.init_model(
            project=NEPTUNE_PROJECT, api_token=NPETUNE_API_TOKEN, with_id=MODEL_ID
        )

        model_versions_df = model.fetch_model_versions_table().to_pandas()

        production_models = model_versions_df[
            model_versions_df["sys/stage"] == "production"
        ]

        for _, model_version in production_models.iterrows():
            version_id = model_version["sys/id"]
            model_version = neptune.init_model_version(
                project=NEPTUNE_PROJECT, api_token=NPETUNE_API_TOKEN, with_id=version_id
            )
            model_version["model"].download("model.joblib")

        model_version.stop()
        model.stop()

        return version_id

    def model_2_tar(self):
        """_summary_."""
        # Build tar file with model data + inference code
        bashCommand = f"tar -cvpzf {self.model_artifacts_tar} model.joblib inference.py"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

    def upload_model_artifact_to_s3(self):
        """_summary_."""
        s3 = self.boto_session.resource("s3")

        # default_bucket = sagemaker.Session().default_bucket()

        # model_artifacts = f"s3://{default_bucket}/{self.model_artifacts_tar}"
        model_artifacts = f"s3://{S3_BUCKET}/{self.model_artifacts_tar}"

        response = s3.meta.client.upload_file(
            self.model_artifacts_tar, S3_BUCKET, self.model_artifacts_tar
        )
        print(response)
        return model_artifacts

    def get_sklearn_image(self):
        """Retrieve sklearn image."""
        image_uri = sagemaker.image_uris.retrieve(
            framework="sklearn",
            region=AWS_REGION,
            version="1.2-1",
            py_version="py3",
            instance_type="ml.m5.xlarge",
        )
        return image_uri

    def create_model(self, model_artifacts, image_uri):
        """_summary_."""
        client = self.sagemaker_client

        model_name = "NYCTAX-RFDV-38" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        create_model_response = client.create_model(
            ModelName=model_name,
            Containers=[
                {
                    "Image": image_uri,
                    "Mode": "SingleModel",
                    "ModelDataUrl": model_artifacts,
                    "Environment": {
                        "SAGEMAKER_SUBMIT_DIRECTORY": model_artifacts,
                        "SAGEMAKER_PROGRAM": "inference.py",
                    },
                }
            ],
            ExecutionRoleArn=AWS_SAGEMAKER_ROLE,
        )
        print("Model Arn: " + create_model_response["ModelArn"])

        return model_name

    def create_endpoint_config(self, model_name):
        """_summary_."""
        client = self.sagemaker_client

        epc_name = "NYCTAX-RFDV-38-epc" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

        client.create_endpoint_config(
            EndpointConfigName=epc_name,
            ProductionVariants=[
                {
                    "VariantName": "sklearnvariant",
                    "ModelName": model_name,
                    # "InstanceType": "ml.c5.large",
                    # "InitialInstanceCount": 1
                    "ServerlessConfig": {"MemorySizeInMB": 1024, "MaxConcurrency": 2},
                },
            ],
        )
        return epc_name

    def create_endpoint(self, epc_name):
        """_summary_."""
        client = self.sagemaker_client

        endpoint_name = "NYCTAX-RFDV-38-ep" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

        create_endpoint_response = client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=epc_name,
        )
        print("Endpoint Arn: " + create_endpoint_response["EndpointArn"])
        return endpoint_name

    def deploy(self):
        """_summary_."""
        model_version = self.get_production_ready_model()

        self.model_2_tar()

        model_artifacts = self.upload_model_artifact_to_s3()
        print(f"model artifacts: {model_artifacts}")

        image_uri = self.get_sklearn_image()

        model_name = self.create_model(model_artifacts, image_uri)
        print(f"model_name: {model_name}")

        epc_name = self.create_endpoint_config(model_name)
        print(f"epc_name: {epc_name}")

        endpoint_name = self.create_endpoint(epc_name)
        print(f"endpoint_name: {endpoint_name}")

        describe_endpoint_response = self.sagemaker_client.describe_endpoint(
            EndpointName=endpoint_name
        )
        while describe_endpoint_response["EndpointStatus"] == "Creating":
            describe_endpoint_response = self.sagemaker_client.describe_endpoint(
                EndpointName=endpoint_name
            )
            print(describe_endpoint_response["EndpointStatus"])
            time.sleep(30)
        print(f"Model endpoint: {endpoint_name}")

        return endpoint_name, model_version

    def infer(self, endpoint_name, test_sample):
        """_summary_."""
        runtime_client = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

        content_type = "application/json"
        request_body = {"Input": test_sample}

        data = json.loads(json.dumps(request_body))
        payload = json.dumps(data)

        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name, ContentType=content_type, Body=payload
        )
        result = json.loads(response["Body"].read().decode())["Output"]

        return result


if __name__ == "__main__":
    sagemaker_client = boto3.client(service_name="sagemaker", region_name=AWS_REGION)
    model_artifacts_tar = "model.tar.gz"
    boto_session = boto3.session.Session()

    deployer = Deployer(sagemaker_client, model_artifacts_tar, boto_session)

    endpoint_name, model_version = deployer.deploy()

    test_sample = {"PULocationID": 9, "DOLocationID": 70, "trip_distance": 20}
    result = deployer.infer(endpoint_name, test_sample)

    report = f"""Deployment Job Report:
                    \nEndpoint: {endpoint_name} deployed correctly.
                    \nModel version deployed from Neptune.AI is: {model_version}.
                    \nEndpoint was tested with {test_sample}.
                    \n\nRMSE:\n{result}\n"""

    # Write metrics to file
    with open("deploy-report.md", "w") as outfile:
        outfile.write(report)
