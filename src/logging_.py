from collections import defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import os

class Logger:
    def __init__(self):
        self.dict = defaultdict(dict) #defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(None))))
        self.setup(0)

    def setup(self, run_id):
        #self.attack_name = attack_name
        #self.model_name = model_name
        #self.hyperparams_str = str(hyperparams)

        self.run_id = run_id

    @property
    def curr_dict(self, value):
        #self.dict[self.attack_name][self.model_name][self.hyperparams_str] = value
        self.dict[self.run_id] = value
    @curr_dict.getter
    def curr_dict(self):
        # return self.dict[self.attack_name][self.model_name][self.hyperparams_str]
        return self.dict[self.run_id]

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
            self.curr_dict[attr_name] = torch.stack(data).numpy()
        else:
            self.curr_dict[attr_name] = np.concatenate((self.curr_dict[attr_name], torch.stack(data).numpy()), axis=1)

            # tmp = self.curr_dict[attr_name]
            #
            # for i, x in enumerate(data):
            #     self.curr_dict[attr_name][i] = torch.cat((tmp[i], data[i]))

    def save(self, run_id=None, force=False, dir="logs"):
        # use default
        if run_id is None:
            run_id = self.run_id

        path = f"{dir}/{run_id}"

        if not force and os.path.exists(path):
            raise RuntimeWarning(run_id)

        # make dir
        os.makedirs(path, exist_ok=True)


        for attr in self.dict[run_id]:
            np.save(f"{dir}/{run_id}/{attr}.npy", self.dict[run_id][attr])

            # with open(path, "wb") as fo:
            #     pickle.dump(self.dict[attack][model][hyperparams_str], fo)

    def save_all(self, force=False, dir="logs"):

        for run_id in self.dict:
            try:
                self.save(run_id, force, dir)
            except RuntimeWarning as e:
                run_id = e.args
                print(f"Log of {run_id} already saved ... skipping")
        #
        # for attack in self.dict:
        #     for model in self.dict[attack]:
        #         for hyperparams_str in self.dict[attack][model]:
        #             try:
        #                 self.save(attack,model, hyperparams_str,force, dir)
        #             except RuntimeWarning as e:
        #                 attack, model, hyperparams_str = e.args
        #                 print(f"Log of {attack}-{model}-{hyperparams_str} already saved ... skipping")

    def load(self, run_id, force=False, dir="logs"):

        if not force and run_id in self.dict:
            raise RuntimeWarning("Record already loaded in dict")

        for entry in os.scandir(f"{dir}/{run_id}"):
            self.dict[run_id][entry.name[:-3]] = np.load(entry.name)

    def load_all(self, force=False, dir="logs"):
        for run_id in os.listdir(dir):
            try:
                self.load(run_id, force=force, dir=dir)
            except RuntimeWarning as e:
                print(f"{run_id} already loaded.")

    def is_logged(self, run_id, dir="logs"):
        path = f"{dir}/{run_id}"
        return os.path.exists(path)

    def plot_progress(self, kind="loss", run_id=None):
        # use default
        if run_id is None:
            run_id = self.run_id

        if kind == "loss":
            plt.plot(self.dict[run_id]["loss_progress"].mean(axis=-1))
        elif kind == "acc":
            plt.plot(self.dict[run_id]["acc_progress"].mean(axis=-1))

    def plot_QD(self, run_id=None):
        # use default
        if run_id is None:
            run_id = self.run_id

        qd = self.QD(self.dict[run_id]["norm_progress"], self.dict[run_id]["acc_progress"])

        plt.plot(qd)

    def QD_at_step(self, step_norm, step_acc, clean_acc):
        # avoid inplace ops
        step_norm_ = step_norm.copy()

        step_norm_[step_acc==1] = np.inf
        step_norm_[clean_acc==0] = 0
        #assert step_acc[clean_acc==0].sum() == 0

        return np.median(step_norm_)

    def QD(self, step_norms, step_accs):
        clean_acc = step_accs[0]
        qd = list()
        for step in range(len(step_norms)):
            qd.append(self.QD_at_step(step_norms[step], step_accs[step], clean_acc))
        return qd
