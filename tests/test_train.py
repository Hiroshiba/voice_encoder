import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy
import pytest
import yaml
from acoustic_feature_extractor.data.sampling_data import SamplingData
from acoustic_feature_extractor.data.wave import Wave
from voice_encoder.trainer import create_trainer
from yaml import SafeLoader

from tests.utility import get_data_directory


@pytest.fixture()
def train_config_path():
    return get_data_directory() / "train_config.yaml"


@pytest.fixture()
def train_config(train_config_path: Path):
    with train_config_path.open() as f:
        return yaml.load(f, SafeLoader)


@pytest.fixture()
def dataset_directory():
    return Path("/tmp/voice_encoder_test_dataset")


def generate_dataset(
    dataset_directory: Path,
    data_num: int,
    sampling_rate: int,
    local_rate: int,
    phoneme_size: int,
    speaker_size: int,
):
    if dataset_directory.exists():
        for p in dataset_directory.rglob("*"):
            if not p.is_dir():
                p.unlink()
    else:
        dataset_directory.mkdir()

    f0_dir = dataset_directory.joinpath("f0")
    phoneme_dir = dataset_directory.joinpath("phoneme")
    wave_dir = dataset_directory.joinpath("wave")
    silence_dir = dataset_directory.joinpath("silence")

    f0_dir.mkdir(exist_ok=True)
    phoneme_dir.mkdir(exist_ok=True)
    wave_dir.mkdir(exist_ok=True)
    silence_dir.mkdir(exist_ok=True)

    for i_data in range(data_num):
        local_length = int(numpy.random.randint(low=100, high=200))
        sampling_length = int(local_length / local_rate * sampling_rate)

        f0 = numpy.random.rand(local_length, 1).astype(numpy.float32)
        f0[f0 < 0.2] = 0
        f0 *= 7
        SamplingData(array=f0, rate=local_rate).save(f0_dir.joinpath(f"{i_data}.npy"))

        phoneme = numpy.random.randint(0, phoneme_size, size=local_length).astype(
            numpy.int32
        )
        phoneme = numpy.identity(phoneme_size)[phoneme].astype(numpy.int32)
        SamplingData(array=phoneme, rate=local_rate).save(
            phoneme_dir.joinpath(f"{i_data}.npy")
        )

        rand = numpy.random.rand()
        wave = numpy.concatenate(
            [
                numpy.sin(
                    (2 * numpy.pi)
                    * (
                        numpy.arange(sampling_length // len(f0), dtype=numpy.float32)
                        * numpy.exp(one_f0)
                        / sampling_rate
                        + rand
                    )
                )
                for one_f0 in f0.tolist()
            ]
        )
        Wave(wave=wave, sampling_rate=sampling_rate).save(
            wave_dir.joinpath(f"{i_data}.wav")
        )

        silence = numpy.zeros_like(wave).astype(bool)
        SamplingData(array=silence, rate=sampling_rate).save(
            silence_dir.joinpath(f"{i_data}.npy")
        )

    speaker_dict = defaultdict(list)
    for i_data in range(data_num):
        speaker_dict[str(i_data % speaker_size)].append(str(i_data))
    json.dump(speaker_dict, dataset_directory.joinpath("speaker_dict.json").open("w"))


def test_train(train_config: Dict[str, Any], dataset_directory: Path):
    generate_dataset(
        dataset_directory=dataset_directory,
        data_num=100,
        sampling_rate=24000,
        local_rate=24000 // numpy.prod(train_config["network"]["scale_list"]),
        phoneme_size=train_config["network"]["phoneme_class_size"],
        speaker_size=train_config["dataset"]["speaker_size"],
    )

    train_config["dataset"]["wave_glob"] = str(dataset_directory.joinpath("wave/*.wav"))
    train_config["dataset"]["silence_glob"] = str(
        dataset_directory.joinpath("silence/*.npy")
    )
    train_config["dataset"]["f0_glob"] = str(dataset_directory.joinpath("f0/*.npy"))
    train_config["dataset"]["phoneme_glob"] = str(
        dataset_directory.joinpath("phoneme/*.npy")
    )
    train_config["dataset"]["speaker_dict_path"] = dataset_directory.joinpath(
        "speaker_dict.json"
    )

    train_config["train"]["batch_size"] = 10
    train_config["train"]["log_iteration"] = 100
    train_config["train"]["snapshot_iteration"] = 500
    train_config["train"]["stop_iteration"] = 1000

    trainer = create_trainer(
        config_dict=train_config, output=Path("/tmp/voice_encoder_test_output")
    )
    trainer.run()
