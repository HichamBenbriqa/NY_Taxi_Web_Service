"""_summary_

"""

from datetime import date
from hydra import initialize, compose


CONFIG_DIR = "../../config"


def get_config(config_path: str = CONFIG_DIR, config_type: str = "data"):
    """
    Read the config file.
    """
    config_file = f"{config_type}.yaml"
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_file)
        if config_type == "data":
            taxi_type = cfg.source.taxi_type
            year = cfg.source.year
            month = cfg.source.month
            return taxi_type, year, month
        if config_type == "model":
            n_estimators = cfg.random_forest_reg.n_estimators
            max_depth = cfg.random_forest_reg.max_depth
            return {"n_estimators": n_estimators, "max_depth": max_depth}
        raise ValueError(f"Invalid config type: {config_type}")


def get_previous_month(year, month):
    """
    Get the previous month and year from the given month and year.
    This method is used to get the test data for the model.
    E.g. if the input month is 1 and input year is 2020, the previous month is 12 and year is 2019.
    The input month and year are the training data. The previous month and year are the test data.
    """
    current_date = date(year, month, 1)
    month, year = (
        (current_date.month - 1, current_date.year)
        if current_date.month != 1
        else (12, current_date.year - 1)
    )

    return year, month
