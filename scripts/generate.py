import argparse
import re
from pathlib import Path
from typing import Optional

import numpy
import yaml
from more_itertools import chunked
from pytorch_trainer.dataset.convert import concat_examples
from tqdm import tqdm
from utility.save_arguments import save_arguments
from voice_encoder.config import Config
from voice_encoder.dataset import SpeakerWavesDataset, WavesDataset, create_dataset
from voice_encoder.generator import Generator


def _extract_number(f):
    s = re.findall(r"\d+", str(f))
    return int(s[-1]) if s else -1


def _get_predictor_model_path(
    model_dir: Path,
    iteration: int = None,
    prefix: str = "predictor_",
):
    if iteration is None:
        paths = model_dir.glob(prefix + "*.pth")
        model_path = list(sorted(paths, key=_extract_number))[-1]
    else:
        model_path = model_dir / (prefix + "{}.pth".format(iteration))
        assert model_path.exists()
    return model_path


def generate(
    model_dir: Path,
    model_iteration: Optional[int],
    model_config: Optional[Path],
    output_dir: Path,
    batch_size: Optional[int],
    num_test: int,
    use_gpu: bool,
):
    if model_config is None:
        model_config = model_dir / "config.yaml"

    output_dir.mkdir(exist_ok=True)
    save_arguments(output_dir / "arguments.yaml", generate, locals())

    config = Config.from_dict(yaml.safe_load(model_config.open()))

    model_path = _get_predictor_model_path(
        model_dir=model_dir,
        iteration=model_iteration,
    )
    generator = Generator(
        config=config,
        predictor=model_path,
        use_gpu=use_gpu,
    )

    dataset = create_dataset(config.dataset)["test"]

    if batch_size is None:
        batch_size = config.train.batch_size

    if isinstance(dataset, SpeakerWavesDataset):
        wave_paths = [data.path_wave for data in dataset.wave_dataset.inputs[:num_test]]
    elif isinstance(dataset, WavesDataset):
        wave_paths = [data.path_wave for data in dataset.inputs[:num_test]]
    else:
        raise Exception()

    for data, wave_path in tqdm(
        zip(chunked(dataset, batch_size), chunked(wave_paths, batch_size)),
        desc="generate",
    ):
        data = concat_examples(data)
        output = generator.generate(wave=data["wave"])

        for feature, p in zip(output, wave_path):
            numpy.save(output_dir / (p.stem + ".npy"), feature)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=Path)
    parser.add_argument("--model_iteration", type=int)
    parser.add_argument("--model_config", type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--use_gpu", action="store_true")
    generate(**vars(parser.parse_args()))
