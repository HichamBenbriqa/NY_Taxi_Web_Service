"""Test cases for the utils.py file."""
import sys

sys.path.append("src/utils")
import utils  # noqa: E402


def test_get_config_data():
    """Test case for the get_config function with config_type = "data"."""
    taxi_type, year, month = utils.get_config(config_type="data")
    assert isinstance(taxi_type, str)
    assert isinstance(year, int)
    assert isinstance(month, int)


def test_get_config_model():
    """Test case for the 'get_config' function with config_type = 'model'."""
    config = utils.get_config(config_type="model")
    assert isinstance(config, dict)
    assert "n_estimators" in config
    assert "max_depth" in config
    assert isinstance(config["n_estimators"], int)
    assert isinstance(config["max_depth"], int)


def test_get_previous_month():
    """Test case for the get_previous_month function."""
    # Test with a month other than January
    year, month = utils.get_previous_month(2022, 5)
    assert year == 2022
    assert month == 4

    # Test with January
    year, month = utils.get_previous_month(2022, 1)
    assert year == 2021
    assert month == 12
