from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from voice_encoder.utility import dataclass_utility
from voice_encoder.utility.git_utility import get_branch_name, get_commit_id


@dataclass
class DatasetConfig:
    input_glob: str
    target_glob: str
    test_num: int
    eval_times_num: int = 1
    seed: int = 0


@dataclass
class NetworkConfig:
    pass


@dataclass
class ModelConfig:
    pass


@dataclass
class TrainConfig:
    batch_size: int
    log_iteration: int
    snapshot_iteration: int
    stop_iteration: int
    num_processes: Optional[int] = None
    optimizer: Dict[str, Any] = field(
        default_factory=dict(
            name="Adam",
        )
    )


@dataclass
class ProjectConfig:
    name: str
    tags: Dict[str, Any] = field(default_factory=dict)


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
    pass
