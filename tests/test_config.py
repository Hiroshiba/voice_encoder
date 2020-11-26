from pathlib import Path

import pytest
import yaml
from voice_encoder.config import Config
from yaml import SafeLoader

from tests.utility import get_data_directory


@pytest.fixture(params=["base_config.yaml", "train_config.yaml"])
def config_path(request):
    return get_data_directory().joinpath(request.param)


def test_from_dict(config_path: Path):
    with config_path.open() as f:
        d = yaml.load(f, SafeLoader)
    Config.from_dict(d)


def test_to_dict(config_path: Path):
    with config_path.open() as f:
        d = yaml.load(f, SafeLoader)
    Config.from_dict(d).to_dict()


def test_equal_base_config_and_reconstructed(config_path: Path):
    with config_path.open() as f:
        d = yaml.load(f, SafeLoader)
    base = Config.from_dict(d)
    base_re = Config.from_dict(base.to_dict())
    assert base == base_re
