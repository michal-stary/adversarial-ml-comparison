import torch
from adv_lib.distances.lp_norms import lp_distances
from adv_lib.utils.losses import difference_of_logits, difference_of_logits_ratio
from torch.nn.functional import cross_entropy
from src.logging_ import Logger
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class PyTorchModelTrackerBase:
        def __init__(self, model, p, n_classes=10, logger=None):
            self._model = model
            self._p = p


            # TODO extract this from model
            self.n_classes= n_classes

            self.logger = Logger() if logger is None else logger

            # self._func_counter = torch.tensor(0)

            # the first value is deemed to be the original input
            # self._tracked_x = []
            # self._tracked_pred = []
            # self._tracked_acc = []
            self.reset()

        def __call__(self, x):
            # compute loss - register forward hook (compute loss, save)
            # compute distance (norm) - register forward hook (compute distance, save)

            # TODO change to track only distances and not really the all images -- too large to fit into RAM
            # self._tracked_x.append(x)  # customize as you want
            with torch.no_grad():
                self._tracked_norm.append(lp_distances(self.inputs, x, p=self._p, dim=1))
                self._func_counter += x.shape[0]


            pred = self._model.__call__(x)

            with torch.no_grad():
                self._tracked_pred.append(pred)
                self._tracked_acc.append((pred.argmax(dim=-1) == self.labels).float())
            return pred

        # call reset after each example
        def reset(self):
            self._func_counter = torch.tensor(0)
            # self._tracked_x = list()
            self._tracked_norm= list()
            self._tracked_pred = list()
            self._tracked_acc = list()

        @property
        def tracked_x(self, value):
            self._tracked_x = value

        @tracked_x.getter
        def tracked_x(self):
            return torch.cat(self._tracked_x)

        @property
        def func_counter(self, value):
            self._func_counter = value

        @func_counter.getter
        def func_counter(self):
            return self._func_counter

        def __getattr__(self, attr):
            """
            Allows to expose interfaces of all existing functions of the
            PyTorch model to the wrapper class.
            """
            try:
                return self.__getattribute__(attr)
            except AttributeError:
                orig_attr = self._model.__getattribute__(attr)
                if callable(orig_attr):
                    def hooked(*args, **kwargs):
                        result = orig_attr(*args, **kwargs)
                        return result

                    return hooked
                else:
                    return orig_attr


class PyTorchModelTrackerSetup(PyTorchModelTrackerBase):
    """
        Tracker class that requires setup call before each forward
    """
    def __init__(self, model, p=float("inf"), loss_f="DL", n_classes=10, logger=None):
        super().__init__(model, p, n_classes, logger)

        if callable(loss_f):
            self._loss_f = loss_f
        else:
            if loss_f == "CE":
                self._loss_f = lambda inputs, labels: cross_entropy(input=inputs, target=labels, reduction='none')
            elif loss_f == "DL":
                self._loss_f = difference_of_logits
            elif loss_f == "DLR":
                self._loss_f = difference_of_logits_ratio
            else:
                raise ValueError("Loss not supported")

    # this function needs to be called before the attack is called
    def setup(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def log(self):
        # log crucial information
        with torch.no_grad():
            self.logger.concat_batch_log("norm_progress",
                        self._tracked_norm)

            self.logger.concat_batch_log("loss_progress",
                        [self._loss_f(pred, self.labels) for pred in self._tracked_pred])

            self.logger.concat_batch_log("acc_progress", self._tracked_acc)

            self.reset()

class PyTorchModelTrackerRecompute(PyTorchModelTrackerBase):
    """
        Tracker class that AIM TO recompute the loss from delta grad at each call
        - no external call is required
        TODO - add backward hook on tensors passed to forward to extract the delta grads
        TODO - recompute the loss from delta grads and inputs somehow
    """
    def __init__(self, model, p, loss_f, logger=None):
        super().__init__(model, p, logger)

    # call reset after each example
    def reset(self):

        # log crucial information
        self.logger.log("norm_progress",
            [lp_distances(self._tracked_x[0], x, p=self._p, dim=1) for x in self._tracked_x])

        # self.logger.log("loss_progress",
        #     [lp_distances(self._tracked_x[0], x, p=self._p, dim=1) for x in self._tracked_x])

        self._func_counter = torch.tensor(0)
        self._tracked_x = list()


if __name__ == '__main__':
    import torchvision

    model = torchvision.models.resnet18(pretrained=True)
    wrapped = PyTorchModelTracker(model)
    inputs = torch.randn((4, 3, 224, 224))
    outs = wrapped(inputs)
    print(wrapped.tracked_x)
    print(wrapped.func_counter)