"""Run the training job to train and evaluate a model for the NY Taxi Web Service."""
from src.data.make_dataset import Data
from src.models.train_model import Trainer
from src.utils.utils import get_config, get_previous_month


def run_training_job():
    """
    Run the training job to train and evaluate a model for the NY Taxi Web Service.

    This function performs the following steps:
    1. Instantiates a Data object for training and testing.
    2. Runs the Data object to download, prepare, and save the train and test data.
    3. Gets the target values for the train and test data to be used for evaluation.
    4. Instantiates a Trainer object to train and evaluate the model.
    5. Saves the pipeline.
    6. Uploads the results to Neptune.
    7. Writes the training job report to a file.
    """
    # Get the taxi_type, year, month from config file.
    taxi_type, year, month = get_config()
    train_data_file = {"taxi_type": taxi_type, "year": year, "month": month}

    # Use previous month data for testing.
    test_year, test_month = get_previous_month(year, month)
    test_data_file = {"taxi_type": taxi_type, "year": test_year, "month": test_month}

    # Instantiate a Data object for training and testing
    train_data = Data(input_data=train_data_file, mode="train")
    test_data = Data(input_data=test_data_file, mode="test")

    # Run the Data object to download, prepare and save the train and test data
    train_data.run()
    test_data.run()

    # Get the target values for the train and test data to be used for evaluation
    y_train, y_test = train_data.get_target_values(), test_data.get_target_values()

    # Instantiate a Trainer object to train and evaluate the model
    params = get_config(config_type="model")
    trainer = Trainer(
        train_data.data_dict,
        y_train,
        test_data.data_dict,
        y_test,
        params=params,
        root_folder="models",
    )
    trainer.train()
    rmse = trainer.evaluate()
    print(trainer.params, rmse)

    # Save the pipeline
    trainer.save_pipeline()
    trainer.upload_to_neptune(rmse)

    report = f"""Training Job Report \nTraining Job parameters:
                    {trainer.params}\nRMSE:\n{rmse}\n"""

    print(report)

    # Write metrics to file
    with open("latest_performance.md", "w") as outfile:
        outfile.write(report)


if __name__ == "__main__":
    run_training_job()
