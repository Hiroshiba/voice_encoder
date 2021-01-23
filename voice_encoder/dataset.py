import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy
from acoustic_feature_extractor.data.sampling_data import SamplingData
from acoustic_feature_extractor.data.wave import Wave
from colorednoise import powerlaw_psd_gaussian
from torch.utils.data import ConcatDataset, Dataset

from voice_encoder.config import DatasetConfig
from voice_encoder.utility.dataset_utility import default_convert


@dataclass
class Input:
    wave: Wave
    silence: SamplingData
    f0: SamplingData
    phoneme: SamplingData


@dataclass
class LazyInput:
    path_wave: Path
    path_silence: Path
    path_f0: Path
    path_phoneme: Path

    def generate(self):
        return Input(
            wave=Wave.load(self.path_wave),
            silence=SamplingData.load(self.path_silence),
            f0=SamplingData.load(self.path_f0),
            phoneme=SamplingData.load(self.path_phoneme),
        )


def mic_augment(wave: numpy.ndarray, sampling_rate: int):
    wave = wave.astype(numpy.float64)

    def make_noise():
        beta = numpy.random.uniform(-3, 3)
        noise = powerlaw_psd_gaussian(
            beta,
            size=len(wave),
            fmin=20 / sampling_rate,
        )
        noise /= 3  # 99.73%が-1~1に入る

        snr = numpy.random.uniform(5, 30)
        noise *= 1 / (10 ** (snr / 20) - 1)
        return noise

    # add noise
    if numpy.random.rand() < 0.9:
        if numpy.random.randint(2, dtype=bool):
            noise = make_noise()
        else:
            noise = (make_noise() + make_noise()) / 2
        wave += noise / 10  # 手加減

    # 音割れ
    if numpy.random.rand() < 0.1:
        th = numpy.random.uniform(0.5, 1)
        wave = numpy.clip(wave, -th, th)

    # 音量
    if numpy.random.rand() < 0.8:
        scale = numpy.random.uniform(0.3, 1.5)
        wave *= scale
        wave = numpy.clip(wave, -1, 1)

    return wave.astype(numpy.float32)


class BaseWaveDataset(Dataset):
    def __init__(
        self,
        sampling_length: int,
        min_not_silence_length: int,
        with_mic_augment: bool,
    ):
        self.sampling_length = sampling_length
        self.min_not_silence_length = min_not_silence_length
        self.with_mic_augment = with_mic_augment

    @staticmethod
    def extract_input(
        sampling_length: int,
        wave_data: Wave,
        silence_data: SamplingData,
        f0_data: SamplingData,
        phoneme_data: SamplingData,
        min_not_silence_length: int,
        with_mic_augment: bool,
    ):
        sr = wave_data.sampling_rate
        sl = sampling_length

        assert len(wave_data.wave) >= sl, f"{len(wave_data.wave)} >= {sl}"

        l_rate = max(f0_data.rate, phoneme_data.rate)

        assert sr % l_rate == 0
        l_scale = int(sr // l_rate)

        local = SamplingData.collect(
            [f0_data, phoneme_data], rate=l_rate, mode="min", error_time_length=0.015
        )
        f0_array = local[:, 0]
        phoneme_array = local[:, 1:]

        length = len(local) * l_scale
        assert (
            abs(length - len(wave_data.wave)) < l_scale * 4
        ), f"{abs(length - len(wave_data.wave))} {l_scale}"

        l_length = length // l_scale
        l_sl = sl // l_scale

        for _ in range(10000):
            if l_length > l_sl:
                l_offset = numpy.random.randint(l_length - l_sl)
            else:
                l_offset = 0
            offset = l_offset * l_scale

            silence = numpy.squeeze(silence_data.resample(sr, index=offset, length=sl))
            if (~silence).sum() >= min_not_silence_length:
                break
        else:
            raise Exception("cannot pick not silence data")

        wave = wave_data.wave[offset : offset + sl]
        f0 = numpy.squeeze(f0_array[l_offset : l_offset + l_sl])
        phoneme = numpy.argmax(phoneme_array[l_offset : l_offset + l_sl], axis=1)

        if with_mic_augment:
            wave = mic_augment(wave, sampling_rate=sr)

        return dict(
            wave=wave,
            f0=f0,
            phoneme=phoneme,
        )

    def make_input(
        self,
        wave_data: Wave,
        silence_data: SamplingData,
        f0_data: SamplingData,
        phoneme_data: SamplingData,
    ):
        return self.extract_input(
            sampling_length=self.sampling_length,
            wave_data=wave_data,
            silence_data=silence_data,
            f0_data=f0_data,
            phoneme_data=phoneme_data,
            min_not_silence_length=self.min_not_silence_length,
            with_mic_augment=self.with_mic_augment,
        )


class WavesDataset(BaseWaveDataset):
    def __init__(
        self,
        inputs: List[Union[Input, LazyInput]],
        sampling_length: int,
        min_not_silence_length: int,
        with_mic_augment: bool,
    ):
        super().__init__(
            sampling_length=sampling_length,
            min_not_silence_length=min_not_silence_length,
            with_mic_augment=with_mic_augment,
        )
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        input = self.inputs[i]
        if isinstance(input, LazyInput):
            input = input.generate()

        return default_convert(
            self.make_input(
                wave_data=input.wave,
                silence_data=input.silence,
                f0_data=input.f0,
                phoneme_data=input.phoneme,
            )
        )


class SpeakerWavesDataset(Dataset):
    def __init__(self, wave_dataset: BaseWaveDataset, speakers: List[int]):
        assert len(wave_dataset) == len(speakers)
        self.wave_dataset = wave_dataset
        self.speakers = speakers

    def __len__(self):
        return len(self.wave_dataset)

    def __getitem__(self, i):
        d = self.wave_dataset[i]
        d["speaker"] = numpy.array(self.speakers[i], dtype=numpy.long)
        return default_convert(d)


def create_dataset(config: DatasetConfig):
    wave_paths = {Path(p).stem: Path(p) for p in glob.glob(str(config.wave_glob))}
    fn_list = sorted(wave_paths.keys())
    assert len(fn_list) > 0

    silence_paths = {Path(p).stem: Path(p) for p in glob.glob(str(config.silence_glob))}
    assert set(fn_list) == set(silence_paths.keys())

    f0_paths = {Path(p).stem: Path(p) for p in glob.glob(str(config.f0_glob))}
    assert set(fn_list) == set(f0_paths.keys())

    phoneme_paths = {Path(p).stem: Path(p) for p in glob.glob(str(config.phoneme_glob))}
    assert set(fn_list) == set(phoneme_paths.keys())

    fn_each_speaker: Dict[str, List[str]] = json.load(open(config.speaker_dict_path))
    assert config.speaker_size == len(fn_each_speaker)

    speakers = {
        fn: speaker
        for speaker, (_, fns) in enumerate(fn_each_speaker.items())
        for fn in fns
    }
    assert set(fn_list).issubset(set(speakers.keys()))

    numpy.random.RandomState(config.seed).shuffle(fn_list)

    num_test = config.num_test
    num_train = (
        config.num_train if config.num_train is not None else len(fn_list) - num_test
    )

    trains = fn_list[num_test:][:num_train]
    tests = fn_list[:num_test]

    def make_dataset(fns, is_train=False, for_evaluate=False):
        inputs = [
            LazyInput(
                path_wave=wave_paths[fn],
                path_silence=silence_paths[fn],
                path_f0=f0_paths[fn],
                path_phoneme=phoneme_paths[fn],
            )
            for fn in fns
        ]

        dataset = WavesDataset(
            inputs=inputs,
            sampling_length=config.sampling_length,
            min_not_silence_length=config.min_not_silence_length,
            with_mic_augment=config.with_mic_augment if is_train else False,
        )

        dataset = SpeakerWavesDataset(
            wave_dataset=dataset,
            speakers=[speakers[fn] for fn in fns],
        )

        if for_evaluate:
            dataset = ConcatDataset([dataset] * config.evaluate_times)

        return dataset

    valid_dataset = (
        create_validation_dataset(config) if config.num_valid is not None else None
    )
    return dict(
        train=make_dataset(trains, is_train=True),
        test=make_dataset(tests),
        eval=make_dataset(tests, for_evaluate=True),
        valid=valid_dataset,
    )


def create_validation_dataset(config: DatasetConfig):
    wave_paths = {Path(p).stem: Path(p) for p in glob.glob(str(config.valid_wave_glob))}
    fn_list = sorted(wave_paths.keys())
    assert len(fn_list) > 0

    silence_paths = {
        Path(p).stem: Path(p) for p in glob.glob(str(config.valid_silence_glob))
    }
    assert set(fn_list) == set(silence_paths.keys())

    f0_paths = {Path(p).stem: Path(p) for p in glob.glob(str(config.valid_f0_glob))}
    assert set(fn_list) == set(f0_paths.keys())

    phoneme_paths = {
        Path(p).stem: Path(p) for p in glob.glob(str(config.valid_phoneme_glob))
    }
    assert set(fn_list) == set(phoneme_paths.keys())

    numpy.random.RandomState(config.seed).shuffle(fn_list)

    valids = fn_list[: config.num_valid]

    inputs = [
        LazyInput(
            path_wave=wave_paths[fn],
            path_silence=silence_paths[fn],
            path_f0=f0_paths[fn],
            path_phoneme=phoneme_paths[fn],
        )
        for fn in valids
    ]

    dataset = WavesDataset(
        inputs=inputs,
        sampling_length=config.sampling_length,
        min_not_silence_length=config.min_not_silence_length,
        with_mic_augment=False,
    )

    dataset = ConcatDataset([dataset] * config.valid_times)
    return dataset
