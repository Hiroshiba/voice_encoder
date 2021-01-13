import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from voice_encoder.utility import dataclass_utility
from voice_encoder.utility.git_utility import get_branch_name, get_commit_id


@dataclass
class DatasetConfig:
    wave_glob: str
    silence_glob: str
    f0_glob: str
    phoneme_glob: str
    speaker_dict_path: Path
    speaker_size: int
    sampling_length: int
    min_not_silence_length: int
    with_mic_augment: bool
    evaluate_times: int
    num_test: int
    num_train: Optional[int] = None
    valid_wave_glob: Optional[str] = None
    valid_silence_glob: Optional[str] = None
    valid_f0_glob: Optional[str] = None
    valid_phoneme_glob: Optional[str] = None
    valid_times: Optional[int] = None
    num_valid: Optional[int] = None
    seed: int = 0


@dataclass
class NetworkConfig:
    hidden_size_list: List[int]
    scale_list: List[int]
    voiced_feature_size: int
    f0_feature_size: int
    phoneme_feature_size: int
    phoneme_class_size: int
    phoneme_subscale_size: Optional[int]
    speaker_size: int


@dataclass
class ModelConfig:
    voiced_loss_weight: float
    f0_loss_weight: float
    phoneme_loss_weight: float


@dataclass
class TrainConfig:
    batch_size: int
    eval_batch_size: Optional[int]
    log_iteration: int
    eval_iteration: int
    stop_iteration: int
    num_processes: Optional[int] = None
    use_multithread: bool = False
    optimizer: Dict[str, Any] = field(default_factory=dict(name="Adam"))


@dataclass
class ProjectConfig:
    name: str
    tags: Dict[str, Any] = field(default_factory=dict)
    category: Optional[str] = None


@dataclass
class Config:
    dataset: DatasetConfig
    network: NetworkConfig
    model: ModelConfig
    train: TrainConfig
    project: ProjectConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        backward_compatible(d)
        return dataclass_utility.convert_from_dict(cls, d)

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_utility.convert_to_dict(self)

    def add_git_info(self):
        self.project.tags["git-commit-id"] = get_commit_id()
        self.project.tags["git-branch-name"] = get_branch_name()


def backward_compatible(d: Dict[str, Any]):
    if "speaker_embedding_size" in d["network"]:
        v = d["network"].pop("speaker_embedding_size")
        if v is not None:
            warnings.warn(f'duplicated d["network"]["speaker_embedding_size"]={v}')

    if "phoneme_subscale_size" not in d["network"]:
        d["network"]["phoneme_subscale_size"] = None

    if "with_mic_augment" not in d["dataset"]:
        d["dataset"]["with_mic_augment"] = False

    if "eval_batch_size" not in d["train"]:
        d["train"]["eval_batch_size"] = None

    if "snapshot_iteration" in d["train"]:
        d["train"]["eval_iteration"] = d["train"].pop("snapshot_iteration")
