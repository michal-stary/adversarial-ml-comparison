from collections import defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import os
import time

class Logger:
    def __init__(self):
        self.dict = defaultdict(dict) #defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(None))))
        self.setup("dataset_model_attack_rest", 0)

    def setup(self, run_id, n_samples):
        self.run_id = run_id
        self.dict[run_id] = dict()
        
        self.single_log("n_samples", n_samples)
        self.single_log("start_time", time.time())
        
        
    @property
    def curr_dict(self, value):
        self.dict[self.run_id] = value

    @curr_dict.getter
    def curr_dict(self):
        return self.dict[self.run_id]

    def single_log(self, attr_name, data):
        self.curr_dict[attr_name] = data
            
    def concat_batch_log(self, attr_name, data):
        if attr_name not in self.curr_dict:
            self.curr_dict[attr_name] = torch.stack(data).to("cpu").numpy()
        else:
            self.curr_dict[attr_name] = np.concatenate((self.curr_dict[attr_name], torch.stack(data).to("cpu").numpy()), axis=1)


    def where(self, **kwargs):
        new = dict()

        for key in self.dict:
            if not all((f"{param}-{kwargs[param]}" in key for param in kwargs)):
                continue
            new[key] = self.dict[key]
        return new


    def save(self, run_id=None, force=False, dir="logs"):
        self.single_log("end_time", time.time())
        
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
            
    def save_all(self, force=False, dir="logs"):

        for run_id in self.dict:
            try:
                self.save(run_id, force, dir)
            except RuntimeWarning as e:
                run_id = e.args
                print(f"Log of {run_id} already saved ... skipping")

    def load(self, run_id, force=False, dir="logs"):

        if not force and run_id in self.dict:
            raise RuntimeWarning("Record already loaded in dict")

        for entry in os.scandir(f"{dir}/{run_id}"):
            self.dict[run_id][entry.name[:-4]] = np.load(f"{dir}/{run_id}/{entry.name}")

    def load_all(self, force=False, dir="logs"):
        for run_id in os.listdir(dir):
            if not os.path.isdir(os.path.join(dir, run_id)):
                continue
            try:
                self.load(run_id, force=force, dir=dir)
            except RuntimeWarning as e:
                print(f"{run_id} already loaded.")

    def is_logged(self, run_id, dir="logs"):
        path = f"{dir}/{run_id}"
        return os.path.exists(path)

    def plot_progress(self, kind="loss", run_id=None):

        # todo deal with not tracked loss/acc
        # - raise warning
        # - add functionality to compute loss,acc from pred + labels

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
        # print(step_norm)
        # avoid inplace ops
        step_norm_ = step_norm.copy()

        step_norm_[step_acc==1] = np.inf
        step_norm_[clean_acc==0] = 0
        #assert step_acc[clean_acc==0].sum() == 0
        # print(step_norm_)
        return np.median(step_norm_)

    def QD(self, step_norms, step_accs):
        clean_acc = step_accs[0]
        qd = list()
        running_min = float("inf")
        for step in range(len(step_norms)):
            curr = self.QD_at_step(step_norms[step], step_accs[step], clean_acc)
            running_min = min(running_min,curr)
            qd.append(running_min)
        return qd
