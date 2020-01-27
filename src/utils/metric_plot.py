from .livelossplot import PlotLosses
import torch


class MetricLivePlot(object):
    def __init__(self):
        self.liveloss = PlotLosses()
        self.__epoch_loss = torch.Tensor()
        self.__epoch_acc = torch.Tensor()
        self.logs = {}
        self.__update_state = {"train": False, "val_": False}

    def update(self, results, train):
        if train:
            prefix = "train"
        else:
            prefix = "val_"

        self.__epoch_loss = results["epoch_loss"]
        self.__epoch_acc = results["epoch_acc"]

        self.logs[prefix + 'log loss'] = self.__epoch_loss.item()
        self.logs[prefix + 'accuracy'] = self.__epoch_acc.item()
        self.__update_state[prefix] = True

    def plot_every(self, iterable):
        for obj in iterable:
            yield obj
            if self.__update_state["train"] and self.__update_state["val_"]:
                self.liveloss.update(self.logs)
                self.liveloss.draw()
                self.reset_states()
            else:
                print("Loss or accuracy has not been update.")
                raise AttributeError

    @property
    def epoch_loss(self):
        return self.__epoch_loss

    @property
    def epoch_acc(self):
        return self.__epoch_acc

    def reset_states(self):
        self.__update_state["train"] = False
        self.__update_state["val_"] = False