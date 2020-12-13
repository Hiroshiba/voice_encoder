from pytorch_trainer.iterators import (
    MultiprocessIterator,
    MultithreadIterator,
    SerialIterator,
)
from torch.utils.data import Dataset


def create_iterator(
    dataset: Dataset,
    batch_size: int,
    for_train: bool = True,
    for_eval: bool = False,
    eval_batch_size: int = None,
    num_processes: int = None,
    use_multithread: bool = False,
):
    if not for_eval or eval_batch_size is None:
        batch_size = batch_size
    else:
        batch_size = eval_batch_size

    if num_processes == 0:
        return SerialIterator(
            dataset,
            batch_size,
            repeat=for_train,
            shuffle=for_train,
        )
    else:
        if not use_multithread:
            return MultiprocessIterator(
                dataset,
                batch_size,
                repeat=for_train,
                shuffle=for_train,
                n_processes=num_processes,
                dataset_timeout=60 * 15,
            )
        else:
            return MultithreadIterator(
                dataset,
                batch_size,
                repeat=for_train,
                shuffle=for_train,
                n_threads=num_processes,
            )
