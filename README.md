# NY_Taxi_Web_Service

This project implements training and deployment pipelines on the open NY Green Taxi dataset: https://www.nyc.gov/site/tlc/businesses/green-cab.page. The project uses various tools and libraries, namely Github Actions, AWS (ECR, SageMaker and S3) and Neptune.ai.

## Getting started

#### Environment variables
You need to create a .env file where you should set the following environment variables:

```
BASE_URL="https://d37ci6vzurychx.cloudfront.net/trip-data/"
S3_BUCKET="YOUR-S3-BUCKET"
AWS_REGION="YOUR-AWS_REGION"
AWS_SAGEMAKER_ROLE="YOUR-AWS_SAGEMAKER_ROLE"
NEPTUNE_PROJECT="YOUR-NPETUNE-PROJECT"
NPETUNE_API_TOKEN="YOUR-NPETUNE-KEY"
MODEL_ID="YOUR-MODEL-ID"
DATA_ROOT_LOCAL_FOLDER="data"
CONFIG_DIR="../../config"
```

You should also set the environment variables AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and AWS_DEFAULT_REGION locally and in Github secrets.

Make sure the SageMaker IAM role has the following permissions: AmazonS3FullAccess, AmazonSageMakerFullAccess

#### Setup

This project uses Python">=3.10,<3.13" and Poetry (https://python-poetry.org/docs/) for dependency management, make sure to install it before executing the next commands:

To install pre-commit and poetry dependencies from the files pyproject.toml and poetry.lock run:

```
make install
```

## More on the project

The is an educational project whose aim is to automate training and deployment pipelines of ML models.

### Training pipeline:

Implemented in the **training_job.py** script:

- Uses the Data class (**src/data/make_dataset.py**) to download, prepare and save the data.
- Instantiates a Trainer object to train and evaluate the model.
- Saves the pipeline.
- Uploads the results to Neptune.
- Writes the training job report to a file.

### Deployment pipeline:

Implemented in the **deployment/deploy.py** script using the Deployer class.

- The Deployer object first gets the production-ready model from Neptune.ai Model registery
- Converts the model file to a .tar, and upload the resulting file to S3
- Create the SageMaker endpoint by:
    - Getting the appropriate sk-learn image from SageMaker
    - Creating the model using the .tar file
    - Creating the SageMaker endpoint configuration
    - Creating the SageMaker endpoint


## MLOPs practises

This project was a chance to implement various MLOPs best practises, namely:

### 1. Portability

To ensure the portability of the project Docker and Amazon Elastic Container Registry (ECR) were used. They were specifically used to containerize the training and the deployment pipelines.

The script training_job.py runs the training pipeline: uses the Data class (**src/data/make_dataset.py**) to download using the NY api, and prepare the train and test datasets. Then, the Trainer class (**src/models/train_model.py**) trains, evaluates, saves and tracks the model/data artifacts. This script is containerized using the Dockerfile in the root folder of the project.

### 2. Traceability

This second objective is achieved using Neptune.ai and AWS S3. Neptune.ai ensures model and data artificats' traceability, while AWS S3 provides a highly available and secure data storage solution.

### 3. Reproducibility

Reproducibility is ensured via the use of containers and the tracking of the data and the hyperparameters used (i.e Neptune.ai)

### 4. Continuous Integration (CI)/Continuous Delivery (CD)

Github Actions automates the process of code integration and deployment. The files **.github/workflows/dev.yml** **.github/workflows/main.yml** implement resepictevly the CI and CD pipelines.

##### Continuous Integration (CI):
This pipeline is triggered whenever there is a push to the dev branch. It consists of two steps: 1- Build the training job's docker image and push it to ECR, 2- Run the docker image and publish the results back using the Continuous Machine Learning (CML) library.

If the results are to their satisfaction, the developer would merge the code to the main branch of the repository.

##### Continuous Delivery (CD):

This pipeline is triggered whenever there is a push to the main branch. It also consists of two steps: 1- Build the deployment job's docker image and push it to ECR, 2- Run the docker image (SageMaker endpoint that serves the model would thus be created), test the endpoint and publish the results back using the Continuous Machine Learning (CML) library.

### 5. Code quality
The code quality is an important aspect every software project. In this project, the library Ruff was used as code linter and formatter.

Git pre-commit hook was also used to run few tests as well as the Ruff linter and formatter before pushing the code to the repository.


## Limitations/Next

#### 1. Feature Store

Feature Stores are a crucial MLOPs component, next I will integrate a feature store (e.g. AWS Feature Store) to save and track better the features produced by the Data class.


#### 2. Model monitoring

Implement some sort of model monitoring using Evidently.AI for example.


#### 3.  More unit tests / Add integration tests

More unit tests need to be implemented.
