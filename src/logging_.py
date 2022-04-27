from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import pickle
import os
import time
from utils.metrics import QD, min_median, n_qs_to_reach, attack_succes_rate


COLORS = {
    "fmn": "red",
    "alma": "blue",
    "apgd": "green"
}


class Logger:
    def __init__(self):
        self.dict = defaultdict(dict) #defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(None))))
        #self.setup("dataset_model_attack_rest", 0)

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

    def value_from_id(self, parameter_name, run_id):
        try:
            return run_id.split(f"{parameter_name}-")[1].split("-")[0]
        except:
            print(run_id)
            
    def row_from_id(self, run_id, index):
        row = dict()
        for par in index[:-1]:
            row[par] = self.value_from_id(par, run_id)
        row["params"] = run_id.split(row["model"])[1]
        return pd.Series(row, index=index)
            
    def where(self, **kwargs):
        new = dict()
        
        for key in self.dict:
            if not all((param not in key or f"{param}-{kwargs[param]}" in key for param in kwargs)):
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
    
    def report_(self, dic=None):
        meds = self.report_medians(dic)
        n_qs = self.report_nqs(dic)
        return pd.merge(meds, n_qs,  how='inner')

    
    def report(self, dic=None):
        if dic is None:
            dic = self.dict
        
        columns = ["dataset", "norm", "model", "attack", "params", "median", "n_qs", "asr"]
        df = pd.DataFrame(columns=columns)
        
        for run_id in dic:
            step_norms = dic[run_id]["norm_progress"]
            step_accs = dic[run_id]["acc_progress"]
            qd = QD(step_norms, step_accs)
            n_qs = n_qs_to_reach(0.1,qd)
            mm = min_median(qd)
            asr = attack_succes_rate(step_accs)

            s = pd.concat([self.row_from_id(run_id, index=columns[:-3]), (pd.Series((mm, n_qs, asr)))])
            s.name = run_id
            s.index= columns
            df = df.append(s)
        return df    

    def report_nqs(self, dic=None):
        if dic is None:
            dic = self.dict
        
        columns = ["dataset", "norm", "model", "attack", "params", "n_qs"]
        df = pd.DataFrame(columns=columns)
        
        for run_id in dic:
            n_qs = n_qs_to_reach(0.1, dic[run_id]["norm_progress"], dic[run_id]["acc_progress"])
            df = df.append(self.row_from_id(run_id, index=columns[:-1]).
                           append(pd.Series(mm, index=[columns[-1]])), ignore_index=True)
        return df    
        
    def report_medians(self, dic=None):
        if dic is None:
            dic = self.dict
        
        columns = ["dataset", "norm", "model", "attack", "params", "median"]
        df = pd.DataFrame(columns=columns)
        
        for run_id in dic:
            mm = min_median(dic[run_id]["norm_progress"], dic[run_id]["acc_progress"])
            df = df.append(self.row_from_id(run_id, index=columns[:-1]).
                           append(pd.Series(mm, index=[columns[-1]])), ignore_index=True)
            # print(df, self.row_from_id(run_id))
            # df.iloc[-1]["median"] = mm
        return df   
            
    def plot_QD(self, run_id=None, ax=None):
        # use default
        if run_id is None:
            run_id = self.run_id

        qd = QD(self.dict[run_id]["norm_progress"], self.dict[run_id]["acc_progress"])

        if ax is None:
            ax = plt.gca()
        attack_name = self.value_from_id("attack", run_id)
        ax.plot(qd, label=attack_name, color=COLORS[attack_name])
        ax.legend(loc='upper left')
        ax.set_xscale("log")
   
    def plot_QD_grid(self, where_settings=dict(), x_axis="model", y_axis="norm", aggregate="best", **kwargs):
        dic = self.where(**where_settings)
        comp = list(dic.keys())
        
        unique_y = list(sorted(set([self.value_from_id(y_axis, c) for c in comp])))
        unique_x_per_y = [list(sorted(set([self.value_from_id(x_axis, c) for c in filter(lambda x: y in x, dic.keys())]))) for y in unique_y]
        
        if aggregate == "best":
            # TODO select best hyperparams for each x,y pair
            for c in comp:
                pass
            
        fig, axs = plt.subplots(len(unique_y), max(map(len, unique_x_per_y)), sharey='none', sharex="row", **kwargs)
        
        # at least 2D
        if len(unique_y) == 1:
            axs = [axs]
        
        for c in comp:
            x_val = self.value_from_id(x_axis, c)
            y_val = self.value_from_id(y_axis, c)
            
            y_index = unique_y.index(y_val)
            

            
            ax = axs[y_index][unique_x_per_y[y_index].index(x_val)]
            self.plot_QD(c, ax=ax)
            
            ax.set_title(x_val)
        plt.show()