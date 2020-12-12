import argparse
import multiprocessing
import warnings
from functools import partial
from pathlib import Path
from typing import Optional

import yaml
from pytorch_trainer.iterators import MultiprocessIterator
from tqdm import tqdm
from voice_encoder.config import Config
from voice_encoder.dataset import create_dataset


def _wrapper(index, dataset):
    try:
        dataset[index]
        return index, None
    except Exception as e:
        return index, e


def _check(dataset, desc: str, num_processes: Optional[int]):
    wrapper = partial(_wrapper, dataset=dataset)

    with multiprocessing.Pool(processes=num_processes) as pool:
        it = pool.imap_unordered(wrapper, range(len(dataset)), chunksize=2 ** 10)
        for i, error in tqdm(it, desc=desc, total=len(dataset)):
            if error is not None:
                print(f"error at {i}")
                breakpoint()


def check_dataset(config_yaml_path: Path):
    with config_yaml_path.open() as f:
        config_dict = yaml.safe_load(f)

    config = Config.from_dict(config_dict)

    warnings.simplefilter("error", MultiprocessIterator.TimeoutWarning)

    num_processes = config.train.num_processes

    # dataset
    datasets = create_dataset(config.dataset)

    _check(datasets["train"], desc="train", num_processes=num_processes)
    _check(datasets["test"], desc="test", num_processes=num_processes)
    _check(datasets["eval"], desc="eval", num_processes=num_processes)

    if datasets["valid"] is not None:
        _check(datasets["valid"], desc="valid", num_processes=num_processes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_yaml_path", type=Path)
    check_dataset(**vars(parser.parse_args()))
