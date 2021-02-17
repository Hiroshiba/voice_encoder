from pytorch_trainer.dataset import convert
from pytorch_trainer.training.updaters import StandardUpdater


class Updater(StandardUpdater):
    def update_core(self):
        iterator = self._iterators["main"]
        batch = iterator.next()
        in_arrays = convert._call_converter(self.converter, batch, self.device)

        optimizer = self._optimizers["main"]
        model = self._models["main"]
        loss_func = self.loss_func or model

        for model in self._models.values():
            model.train()
        optimizer.zero_grad()

        if isinstance(in_arrays, tuple):
            loss = loss_func(*in_arrays)
        elif isinstance(in_arrays, dict):
            loss = loss_func(**in_arrays)
        else:
            loss = loss_func(in_arrays)

        loss.backward()
        optimizer.step()
