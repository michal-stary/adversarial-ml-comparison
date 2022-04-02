from collections import defaultdict
import torch
import matplotlib.pyplot as plt
import json
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

    def save(self, dir="logs", attack=None, model=None, hyperparams_str=None):
        # use default
        if attack is None:
            attack = self.attack_name
            model = self.model_name
            hyperparams_str =self.hyperparams_str

        # create json object from dictionary
        json_ = json.dumps(self.dict[attack][model][hyperparams_str])

        # open file for writing, "w"
        with open(f"{dir}/{attack}-{model}-{hyperparams_str}.json", "w") as fo:
            # write json object to file
            fo.write(json_)

    def save_all(self):
        for attack in self.dict:
            for model in self.dict[attack]:
                for hyperparams_str in self.dict[attack][model]:
                    self.save(attack,model, hyperparams_str)

    def load(self, attack, model, hyperparams_str, dir="logs"):

        if hyperparams_str in self.dict[attack][model]:
            raise ValueError("Record already loaded in dict")

        with open(f"{dir}/{attack}-{model}-{hyperparams_str}.json", "r") as f:
            self.dict[attack][model][hyperparams_str] = json.load(f)

    def load_all(self, dir="logs"):
        for filename in os.listdir(dir):
            attack, model, hyperparams_str = filename.split(".json")[0].split("-")
            try:
                self.load(attack, model, hyperparams_str, dir=dir)
            except ValueError as e:
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

        # TODO