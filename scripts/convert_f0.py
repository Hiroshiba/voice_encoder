import argparse
from glob import glob
from pathlib import Path

import numpy
import yaml
from acoustic_feature_extractor.data.sampling_data import SamplingData
from tqdm import tqdm
from utility.save_arguments import save_arguments
from voice_encoder.config import Config


def convert_f0(
    model_config: Path,
    input_glob: str,
    input_f0_statistics: Path,
    target_f0_statistics: Path,
    output_dir: Path,
):
    output_dir.mkdir(exist_ok=True)
    save_arguments(output_dir / "arguments.yaml", convert_f0, locals())

    config = Config.from_dict(yaml.safe_load(model_config.open()))

    input_stat = numpy.load(input_f0_statistics, allow_pickle=True).item()
    target_stat = numpy.load(target_f0_statistics, allow_pickle=True).item()

    paths = list(map(Path, glob(input_glob)))

    for p in tqdm(paths, desc="convert_f0"):
        data = SamplingData.load(p)

        if data.array.shape[1] == (
            config.network.voiced_feature_size + 1 + config.network.phoneme_feature_size
        ):
            f0_index = config.network.voiced_feature_size
        elif data.array.shape[1] == (1 + 1 + 40):
            f0_index = 1
        else:
            raise ValueError(data.array.shape[1])

        data.array[:, f0_index] += target_stat["mean"] - input_stat["mean"]
        data.save(output_dir / (p.stem + ".npy"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", required=True, type=Path)
    parser.add_argument("--input_glob", required=True)
    parser.add_argument("--input_f0_statistics", required=True, type=Path)
    parser.add_argument("--target_f0_statistics", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    convert_f0(**vars(parser.parse_args()))
