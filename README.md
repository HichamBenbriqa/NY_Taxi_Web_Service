# NY_Taxi_Web_Service

This project implements training and deployment pipelines on the open NY Green Taxi dataset: https://www.nyc.gov/site/tlc/businesses/green-cab.page. The project uses various tools and libraries, namely Github Actions, AWS (ECR, SageMaker and S3) and Neptune.ai.

The is an educational project, and the aim is to build automated training and deployment pipelines are implemented while respecting MLOPs best practises.

## Training pipeline:

Implemented in the **training_job.py** script:

- Uses the Data class (**src/data/make_dataset.py**) to download, prepare and save the data.
- Instantiates a Trainer object to train and evaluate the model.
- Saves the pipeline.
- Uploads the results to Neptune.
- Writes the training job report to a file.

## Deployment pipeline:

Implemented in the **deployment/deploy.py** script using the Deployer class.

- The Deployer object first gets the production-ready model from Neptune.ai Model registery
- Converts the model file to a .tar, and upload the resulting file to S3
- Create the SageMaker endpoint by:
    - Getting the appropriate sk-learn image from SageMaker
    - Creating the model using the .tar file
    - Creating the SageMaker endpoint configuration
    - Creating the SageMaker endpoint


Various MLOPs best practises were followed, namely:

### 1. Portability

To ensure the portability of the project Docker and Amazon Elastic Container Registry (ECR) were used. Specifically, Docker and ECR are used to containerize the training and the deployment pipelines.

The script training_job.py runs the training pipeline: uses the Data class (**src/data/make_dataset.py**) to download using the NY api, and prepare the train and test datasets. Then, the Trainer class (**src/models/train_model.py**) trains, evaluates, saves and tracks the model/data artifacts.

### 2. Tracability

This second objective is achieved using Neptune.ai and AWS S3. Neptune.ai ensures model and data artificats' tracebility, while AWS S3 provides a highly available and secure data storage solution.

### 3. Reproducibility

Reproducibility is ensured via the use of containers and the tracking of the data and the hyperparameters used (i.e Neptune.ai)

### 4. Continuous Integration (CI)/Continuous Delivery (CD)

Using Github Actions I managed to automate the process of code integration and deployment. Indeed, the files .github/workflows/dev.yml .github/workflows/main.yml implement resepictevly the CI and CD pipelines.

##### Continuous Integration (CI):
This pipeline is triggered whenever there is a push to the dev branch. It consists of two steps: 1- Build the training job's docker image and push it to ECR, 2- Run the docker image and publish the results back using the Continuous Machine Learning (CML) library.

If the results are to their satisfaction, the developer would merge the code to the main branch of the repository.

##### Continuous Delivery (CD):

This pipeline is triggered whenever there is a push to the main branch. It also consists of two steps: 1- Build the deployment job's docker image and push it to ECR, 2- Run the docker image (SageMaker endpoint that serves the model would thus be created), test the endpoint and publish the results back using the Continuous Machine Learning (CML) library.

### 5. Code quality


### 6. Dependency management


## Project Structure
.
├── config/
├── data/
├── deployment/
├── models/
├── notebooks/
├── src
│   ├── data/
│   ├── models/
│   └── utils/
├── tests/
├── Dockerfile
├── Makefile
├── poetry.lock
├── pyproject.toml
├── README.md
├── run_commands.py
└── training_job.py
