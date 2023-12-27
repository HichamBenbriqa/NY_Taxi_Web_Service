"""Unit tests of the Trainer class methods."""
import os

import pytest
from sklearn.datasets import make_regression

from models.train_model import Trainer


@pytest.fixture
def trainer():
    """
    Trains a model using dummy data for testing purposes.

    Returns
        Trainer: The trained model.
    """
    # Generate some dummy data for testing
    X_train, y_train = make_regression(n_samples=100, n_features=10, random_state=42)
    X_test, y_test = make_regression(n_samples=50, n_features=10, random_state=42)
    dict_train = [dict(zip(range(10), row)) for row in X_train]  # noqa: B905
    dict_test = [dict(zip(range(10), row)) for row in X_test]  # noqa: B905
    params = {"n_estimators": 100, "max_depth": 5}
    root_folder = "test_models"
    trainer = Trainer(
        dict_train=dict_train,
        y_train=y_train,
        dict_test=dict_test,
        y_test=y_test,
        params=params,
        root_folder=root_folder,
    )
    yield trainer
    # Clean up the test models folder after each test
    if os.path.exists(root_folder):
        for file_name in os.listdir(root_folder):
            file_path = os.path.join(root_folder, file_name)
            os.remove(file_path)
        os.rmdir(root_folder)


def test_train(trainer):
    """
    Test the train method of the trainer object.

    Args:
        trainer: The trainer object to test.

    Returns:
        None
    """
    trainer.train()
    assert trainer.pipeline is not None


def test_evaluate(trainer):
    """
    Test the evaluate method of the trainer object.

    Args:
        trainer: The trainer object.

    Returns:
        None
    """
    trainer.train()
    rmse = trainer.evaluate()
    assert isinstance(rmse, float)


def test_predict(trainer):
    """
    Test the predict method of the trainer object.

    Args:
        trainer: The trainer object used for training and prediction.

    Returns:
        None
    """
    trainer.train()
    features = {
        "feature_1": 1,
        "feature_2": 2,
        "feature_3": 3,
        "feature_4": 4,
        "feature_5": 5,
        "feature_6": 6,
        "feature_7": 7,
        "feature_8": 8,
        "feature_9": 9,
        "feature_10": 10,
    }
    prediction = trainer.predict(features)
    assert isinstance(prediction, float)


def test_save_load_pipeline(trainer):
    """
    Test case to verify the functionality of saving and loading the pipeline.

    Args:
        trainer: The trainer object used for training the model.

    Raises:
        AssertionError: If the pipeline path does not exist or if the loaded pipeline is
        None.
    """
    trainer.train()
    trainer.save_pipeline()
    assert os.path.exists(trainer.pipeline_path)
    trainer.load_pipeline()
    assert trainer.pipeline is not None


def test_upload_to_neptune(trainer):
    """
    Test the upload_to_neptune method of the trainer object.

    This function trains the model, evaluates its performance, and then attempts to
    upload the evaluation result to Neptune.
    It asserts that an exception is raised during the upload process.

    Args:
        trainer: The trainer object used for training and evaluation.
    """
    trainer.train()
    rmse = trainer.evaluate()
    with pytest.raises(Exception):  # noqa: B017
        trainer.upload_to_neptune(rmse)
