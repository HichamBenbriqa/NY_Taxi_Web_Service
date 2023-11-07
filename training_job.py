"""_summary_

"""

import sys

from src.data.make_dataset import Data
from src.models.train_model import Trainer
from src.utils.utils import get_config, get_previous_month

## Instantiate a Data object for training and testing
train_data = Data()
test_data = Data("green", year= "2019", month="12", mode="test")

## Run the Data object to download, prepare and save the train and test data
train_data.run()
test_data.run()

## Get the target values for the train and test data to be used for evaluation
y_train, y_test = train_data.get_target_values(), test_data.get_target_values()

## Instantiate a Trainer object to train and evaluate the model
params = get_config(config_type="model")
trainer = Trainer(train_data.data_dict, y_train, test_data.data_dict, y_test, params=params, root_folder="models")
trainer.train()
rmse = trainer.evaluate()
print(trainer.params, rmse)

## Save the pipeline
trainer.save_pipeline()


report = (f" Training Job Submission Report\n\n"
           f"Training Job parameters: {trainer.params}\n\n"
            "RMSE:\n\n"
           f"{rmse}\n\n"
          )
print(report)
# Write metrics to file
with open('report.txt', 'w') as outfile:
    outfile.write(report)
