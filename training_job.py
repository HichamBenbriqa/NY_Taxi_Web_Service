"""_summary_."""

from src.data.make_dataset import Data
from src.models.train_model import Trainer
from src.utils.utils import get_config

## Instantiate a Data object for training and testing
train_data_file = {"taxi_type": "green", "year": 2021, "month": 12}
test_data_file = {"taxi_type": "green", "year": 2021, "month": 11}

train_data = Data(input_data=train_data_file, mode="train")
test_data = Data(input_data=test_data_file, mode="test")

## Run the Data object to download, prepare and save the train and test data
train_data.run()
test_data.run()

## Get the target values for the train and test data to be used for evaluation
y_train, y_test = train_data.get_target_values(), test_data.get_target_values()

## Instantiate a Trainer object to train and evaluate the model
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

## Save the pipeline
trainer.save_pipeline()
trainer.upload_to_neptune(rmse)


report = (
    f" Training Job Report \nTraining Job parameters: {trainer.params}\nRMSE:\n{rmse}\n"
)

print(report)

# Write metrics to file
with open("report.md", "w") as outfile:
    outfile.write(report)
