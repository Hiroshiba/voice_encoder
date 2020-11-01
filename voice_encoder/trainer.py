import warnings
from copy import copy
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from pytorch_trainer.iterators import MultiprocessIterator
from pytorch_trainer.training import Trainer, extensions
from pytorch_trainer.training.updaters import StandardUpdater
from ranger import Ranger
from tensorboardX import SummaryWriter
from torch import optim
from torch.optim.optimizer import Optimizer

from voice_encoder.config import Config
from voice_encoder.dataset import create_dataset
from voice_encoder.model import Model, create_network
from voice_encoder.utility.pytorch_utility import init_orthogonal
from voice_encoder.utility.trainer_extension import TensorboardReport, WandbReport


def create_trainer(
    config_dict: Dict[str, Any],
    output: Path,
):
    # config
    config = Config.from_dict(config_dict)
    config.add_git_info()

    output.mkdir(parents=True)
    with (output / "config.yaml").open(mode="w") as f:
        yaml.safe_dump(config.to_dict(), f)

    # model
    networks = create_network(config.network)
    model = Model(config=config.model, networks=networks)
    init_orthogonal(model)

    device = torch.device("cuda")
    model.to(device)

    # dataset
    def _create_iterator(dataset, for_train: bool):
        return MultiprocessIterator(
            dataset,
            config.train.batch_size,
            repeat=for_train,
            shuffle=for_train,
            n_processes=config.train.num_processes,
            dataset_timeout=60 * 15,
        )

    datasets = create_dataset(config.dataset)
    train_iter = _create_iterator(datasets["train"], for_train=True)
    test_iter = _create_iterator(datasets["test"], for_train=False)
    # eval_iter = _create_iterator(datasets["eval"], for_train=False)

    warnings.simplefilter("error", MultiprocessIterator.TimeoutWarning)

    # optimizer
    cp: Dict[str, Any] = copy(config.train.optimizer)
    n = cp.pop("name").lower()

    optimizer: Optimizer
    if n == "adam":
        optimizer = optim.Adam(model.parameters(), **cp)
    elif n == "sgd":
        optimizer = optim.SGD(model.parameters(), **cp)
    elif n == "ranger":
        optimizer = Ranger(model.parameters(), **cp)
    else:
        raise ValueError(n)

    # updater
    updater = StandardUpdater(
        iterator=train_iter,
        optimizer=optimizer,
        model=model,
        device=device,
    )

    # trainer
    trigger_log = (config.train.log_iteration, "iteration")
    trigger_snapshot = (config.train.snapshot_iteration, "iteration")
    trigger_stop = (
        (config.train.stop_iteration, "iteration")
        if config.train.stop_iteration is not None
        else None
    )

    trainer = Trainer(updater, stop_trigger=trigger_stop, out=output)

    ext = extensions.Evaluator(test_iter, model, device=device)
    trainer.extend(ext, name="test", trigger=trigger_log)

    ext = extensions.snapshot_object(
        networks.predictor, filename="predictor_{.updater.iteration}.pth"
    )
    trainer.extend(ext, trigger=trigger_snapshot)

    trainer.extend(extensions.FailOnNonNumber(), trigger=trigger_log)
    trainer.extend(extensions.LogReport(trigger=trigger_log))
    trainer.extend(
        extensions.PrintReport(["iteration", "main/loss", "test/main/loss"]),
        trigger=trigger_log,
    )

    ext = TensorboardReport(writer=SummaryWriter(Path(output)))
    trainer.extend(ext, trigger=trigger_log)

    if config.project.category is not None:
        ext = WandbReport(
            config_dict=config.to_dict(),
            project_category=config.project.category,
            project_name=config.project.name,
            output_dir=output.joinpath("wandb"),
        )
        trainer.extend(ext, trigger=trigger_log)

    (output / "struct.txt").write_text(repr(model))

    if trigger_stop is not None:
        trainer.extend(extensions.ProgressBar(trigger_stop))

    return trainer
