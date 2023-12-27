"""_summary_."""

import argparse
import sys

from src.data.make_dataset import Data
from src.models.train_model import Trainer
from src.utils.utils import get_config, get_previous_month


def init_arg_parser():
    """
    Initialize the argument parser.

    :return: an ArgumentParser with its arguments loaded
    :rtype: argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(description="Train the model")
    p.add_argument(
        "-t",
        "--train",
        dest="train",
        action="store_true",
        help="Train and save a model from data",
    )
    p.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="Evaluate trained model on test data",
    )
    p.add_argument("-tt", "--taxi_type", dest="taxi_type")
    p.add_argument("-y", "--year", dest="year")
    p.add_argument("-m", "--month", dest="month")

    return p


if __name__ == "__main__":
    parser = init_arg_parser()
    args = parser.parse_args()
    taxi_type, year, month = args.taxi_type, int(args.year), int(args.month)

    if args.train:
        test_year, test_month = get_previous_month(year, month)

        ## Instantiate a Data object for training and testing
        train_data = Data(taxi_type, year, month, mode="train", root_folder="data")
        test_data = Data(
            taxi_type, test_year, test_month, mode="test", root_folder="data"
        )

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
    elif args.evaluate:
        test_data = Data(taxi_type, year, month, mode="test", root_folder="data")
        test_data.run()
        y_test = test_data.get_target_values()

        evaluater = Trainer(
            dict_test=test_data.data_dict, y_test=y_test, root_folder="models"
        )

        rmse = evaluater.evaluate()
        print(rmse)
    else:
        parser.print_help()
        sys.exit(1)
