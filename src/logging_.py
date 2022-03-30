from collections import defaultdict
import torch
import matplotlib.pyplot as plt

class Logger:
    def __init__(self):
        self.dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(None))))

        self.setup("model", "attack", dict())

    def setup(self, model_name, attack_name, hyperparams):
        self.attack_name = model_name
        self.model_name = attack_name
        self.hyperparams_str = str(hyperparams)
        # self.dict[self.attack_name][self.model_name][self.hyperparams_str] = defaultdict(None)
        # self.dict[self.attack_name][self.model_name][self.hyperparams_str]

    def log(self, attr_name, data):
        self.dict[self.attack_name][self.model_name][self.hyperparams_str][attr_name] = data

    def sum_log(self, attr_name, data):
        self.dict[self.attack_name][self.model_name][self.hyperparams_str][attr_name] += data

    def append_log(self, attr_name, data):
        if self.dict[self.attack_name][self.model_name][self.hyperparams_str][attr_name] is None:
            self.dict[self.attack_name][self.model_name][self.hyperparams_str][attr_name] = [data]
        else:
            self.dict[self.attack_name][self.model_name][self.hyperparams_str][attr_name].append(data)

    def extend_log(self, attr_name, data):
        if self.dict[self.attack_name][self.model_name][self.hyperparams_str][attr_name] is None:
            self.dict[self.attack_name][self.model_name][self.hyperparams_str][attr_name] = data
        else:
            self.dict[self.attack_name][self.model_name][self.hyperparams_str][attr_name] += data

    def concat_batch_log(self, attr_name, data):
        """
        Enlarge the list of tensors in attr_name by concating new batch tensor on each corresponding location of top
        level list

        Example:
        dict_state: [Tensor(2,0), Tensor(3,9)],
        data: [Tensor([39,20]), Tensor([28,6])]
            -> [Tensor([2,0,39,20]), Tensor([3,9,28,6])]

        :param attr_name:
        :param data: List of Tensors
        :return: None
        """
        if attr_name not in self.dict[self.attack_name][self.model_name][self.hyperparams_str]:
            self.dict[self.attack_name][self.model_name][self.hyperparams_str][attr_name] = data
        else:
            tmp = self.dict[self.attack_name][self.model_name][self.hyperparams_str][attr_name]

            for i, x in enumerate(data):
                self.dict[self.attack_name][self.model_name][self.hyperparams_str][attr_name][i] = torch.cat((tmp[i], data[i]))

    def __getattr__(self, attr):
        try:
            return self.__getattribute__(attr)
        except AttributeError:
            return self.dict[self.attack_name][self.model_name][self.hyperparams_str][attr]




    def plot_progress(self, kind="loss", attack=None, model=None, hyperparams_str=None):
        # use default
        if attack is None:
            attack = self.attack_name
            model = self.model_name
            hyperparams_str =self.hyperparams_str

        if kind == "loss":
            plt.plot([x.mean().detach().numpy() for x in self.dict[attack][model][hyperparams_str]["loss_progress"]])
        elif kind == "acc":
            plt.plot([x.mean().detach().numpy() for x in self.dict[attack][model][hyperparams_str]["acc_progress"]])

    def plot_QD(self, attack=None, model=None, hyperparams_str=None):
        # use default
        if attack is None:
            attack = self.attack_name
            model = self.model_name
            hyperparams_str =self.hyperparams_str

        # TODO