import argparse
import re
from glob import glob
from pathlib import Path
from typing import Optional

import numpy
import torch
import yaml
from acoustic_feature_extractor.data.sampling_data import SamplingData
from acoustic_feature_extractor.data.wave import Wave
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


def _get_model_path(
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
    to_f0_scaler: bool,
    batch_size: Optional[int],
    num_test: int,
    target_glob: Optional[str],
    use_gpu: bool,
):
    if model_config is None:
        model_config = model_dir / "config.yaml"

    output_dir.mkdir(exist_ok=True)
    save_arguments(output_dir / "arguments.yaml", generate, locals())

    config = Config.from_dict(yaml.safe_load(model_config.open()))

    generator = Generator(
        config=config,
        predictor=_get_model_path(
            model_dir=model_dir,
            iteration=model_iteration,
            prefix="predictor_",
        ),
        f0_network=(
            None
            if not to_f0_scaler
            else _get_model_path(
                model_dir=model_dir,
                iteration=model_iteration,
                prefix="f0_network_",
            )
        ),
        use_gpu=use_gpu,
    )

    dataset = create_dataset(config.dataset)["test"]
    scale = numpy.prod(config.network.scale_list)

    if batch_size is None:
        batch_size = config.train.batch_size

    if isinstance(dataset, SpeakerWavesDataset):
        wave_paths = [data.path_wave for data in dataset.wave_dataset.inputs[:num_test]]
    elif isinstance(dataset, WavesDataset):
        wave_paths = [data.path_wave for data in dataset.inputs[:num_test]]
    else:
        raise Exception()

    if target_glob is not None:
        wave_paths += list(map(Path, glob(target_glob)))

    for wps in tqdm(chunked(wave_paths, batch_size), desc="generate"):
        waves = [Wave.load(p) for p in wps]
        arrays = [w.wave for w in waves]

        pad_lengths = [int(numpy.ceil(len(w) / scale) * scale) for w in arrays]
        arrays = [numpy.r_[w, numpy.zeros(max(pad_lengths) - len(w))] for w in arrays]

        tensors = [torch.from_numpy(array.astype(numpy.float32)) for array in arrays]
        output = generator.generate(
            wave=concat_examples(tensors), to_f0_scaler=to_f0_scaler
        )

        for feature, p, w, l in zip(output, wps, waves, pad_lengths):
            feature = feature.T[: l // scale]
            data = SamplingData(array=feature, rate=w.sampling_rate // scale)
            data.save(output_dir / (p.stem + ".npy"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=Path)
    parser.add_argument("--model_iteration", type=int)
    parser.add_argument("--model_config", type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--to_f0_scaler", action="store_true")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_test", type=int, default=10)
    parser.add_argument("--target_glob")
    parser.add_argument("--use_gpu", action="store_true")
    generate(**vars(parser.parse_args()))
