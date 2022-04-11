import torch
from adv_lib.distances.lp_norms import lp_distances
from adv_lib.utils.losses import difference_of_logits, difference_of_logits_ratio
from torch.nn.functional import cross_entropy
from src.logging_ import Logger


class PyTorchModelTracker:
        def __init__(self, model, p, logger=None, loss_f=None, track_acc=False):
            self._model = model
            self._p = p
            self.logger = Logger() if logger is None else logger
            self.track_acc = track_acc
            self.reset()

            if callable(loss_f):
                self._loss_f = loss_f
            else:
                if loss_f == "CE":
                    self._loss_f = lambda inputs, labels: cross_entropy(input=inputs, target=labels, reduction='none')
                elif loss_f == "DL":
                    self._loss_f = difference_of_logits
                elif loss_f == "DLR":
                    self._loss_f = difference_of_logits_ratio
                elif loss_f is None:
                    self._loss_f = None

                else:
                    raise ValueError("Loss not supported")


        # this function needs to be called before the attack is called
        def setup(self, inputs, labels):
            self.inputs = inputs
            self.labels = labels

        def __call__(self, x):
            with torch.no_grad():
                self._tracked_norm.append(lp_distances(self.inputs, x, p=self._p, dim=1))
                self._func_counter += x.shape[0]

            pred = self._model.__call__(x)

            with torch.no_grad():
                self._tracked_pred.append(pred.detach())
                # self._tracked_labels.append(self.labels.detach())

                if self._loss_f is not None:
                    self._tracked_loss.append(self._loss_f(pred, self.labels))
                if self.track_acc:
                    self._tracked_acc.append((pred.argmax(dim=-1) == self.labels).float())
            return pred

        # call reset after each attack call
        def reset(self):
            self._func_counter = torch.tensor(0)
            self._tracked_norm= list()
            self._tracked_loss= list()
            self._tracked_acc = list()
            self._tracked_pred = list()
            # self._tracked_labels = list()

        def log(self):
            # log crucial information
            with torch.no_grad():
                self.logger.concat_batch_log("norm_progress",
                                             self._tracked_norm)

                self.logger.concat_batch_log("pred_progress",
                                             self._tracked_pred)
                self.logger.concat_batch_log("labels_progress",
                                            [self.labels])

                if self._loss_f is not None:
                    self.logger.concat_batch_log("loss_progress",
                                                 self._tracked_loss)
                if self.track_acc:
                    self.logger.concat_batch_log("acc_progress", self._tracked_acc)

                self.reset()

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





