from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import os

class Logger:
    def __init__(self):
        self.dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(None))))

        self.setup("attack", "model", dict())

    def setup(self, attack_name, model_name, hyperparams):
        self.attack_name = attack_name
        self.model_name = model_name
        self.hyperparams_str = str(hyperparams)

    @property
    def curr_dict(self, value):
        self.dict[self.attack_name][self.model_name][self.hyperparams_str] = value

    @curr_dict.getter
    def curr_dict(self):
        return self.dict[self.attack_name][self.model_name][self.hyperparams_str]

    def log(self, attr_name, data):
        self.curr_dict[attr_name] = data

    def sum_log(self, attr_name, data):
        self.curr_dict[attr_name] += data

    def append_log(self, attr_name, data):
        if self.curr_dict[attr_name] is None:
            self.curr_dict[attr_name] = [data]
        else:
            self.curr_dict[attr_name].append(data)

    def extend_log(self, attr_name, data):
        if self.curr_dict[attr_name] is None:
            self.curr_dict[attr_name] = data
        else:
            self.curr_dict[attr_name] += data

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
        if attr_name not in self.curr_dict:
            self.curr_dict[attr_name] = data
        else:
            tmp = self.curr_dict[attr_name]

            for i, x in enumerate(data):
                self.curr_dict[attr_name][i] = torch.cat((tmp[i], data[i]))

    def __getattr__(self, attr):
        try:
            return self.__getattribute__(attr)
        except AttributeError:
            return self.curr_dict[attr]

    def save(self, attack=None, model=None, hyperparams_str=None, force=False, dir="logs"):
        # use default
        if attack is None:
            attack = self.attack_name
            model = self.model_name
            hyperparams_str =self.hyperparams_str

        path = f"{dir}/{attack}-{model}-{hyperparams_str}"

        if not force and os.path.exists(path):
            raise RuntimeWarning(attack, model, hyperparams_str)

        with open(path, "wb") as fo:
            pickle.dump(self.dict[attack][model][hyperparams_str], fo)

    def save_all(self, force=False, dir="logs"):
        for attack in self.dict:
            for model in self.dict[attack]:
                for hyperparams_str in self.dict[attack][model]:
                    try:
                        self.save(attack,model, hyperparams_str,force, dir)
                    except RuntimeWarning as e:
                        attack, model, hyperparams_str = e.args
                        print(f"Log of {attack}-{model}-{hyperparams_str} already saved ... skipping")

    def load(self, attack, model, hyperparams_str, force=False, dir="logs"):

        if not force and hyperparams_str in self.dict[attack][model]:
            raise RuntimeWarning("Record already loaded in dict")

        with open(f"{dir}/{attack}-{model}-{hyperparams_str}", "rb") as f:
            self.dict[attack][model][hyperparams_str] = pickle.load(f)

    def load_all(self, force=False, dir="logs"):
        for filename in os.listdir(dir):
            attack, model, hyperparams_str = filename.split("-")
            try:
                self.load(attack, model, hyperparams_str, force=force, dir=dir)
            except RuntimeWarning as e:
                print(f"{attack}-{model}-{hyperparams_str} already loaded.")


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


        qd = self.QD(self.dict[attack][model][hyperparams_str]["norm_progress"], self.dict[attack][model][hyperparams_str]["acc_progress"])

        plt.plot(qd)

    def QD_at_step(self, step_norm, step_acc, clean_acc):
        # avoid inplace ops
        step_norm_ = step_norm.copy()

        step_norm_[step_acc==1] = torch.inf
        step_norm_[clean_acc==0] = 0
        assert step_acc[clean_acc==0].sum() == 0

        return torch.median(step_norm_)

    def QD(self, step_norms, step_accs):
        clean_acc = step_accs[0]
        qd = list()
        for step in range(len(step_norms)):
            qd.append(self.QD_at_step(step_norms[step], step_accs[step], clean_acc))
        return qd
