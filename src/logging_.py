from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

import pickle
import os
import time
from copy import deepcopy
from utils.metrics import QD, SE, min_median, n_qs_to_reach, attack_succes_rate, min_norms, adversify,\
                            steps_to_find_adv, first_to_final_ratio, min_norms_pred
from scores import THRESHOLDS, SCORES

from matplotlib.colors import Normalize

NORM10 = Normalize(0,9)
COLORS = {
    "fmn": "red",
    "alma": "blue",
    "apgd": "green",
    "ddn": "black",
    "ensemble": "orange",
    "pdgd": "purple",
    "pdpgd": "purple",
    "aa": "brown",
    "afm": "blue",
    "fmn_t": "red",
    "alma_t": "blue",
    "fmn_all": "orange",
    "alma_all": "purple",
    
    
    "0.05": "red",
    "0.03": "green",
    "0.01": "blue",
    
    "0.1": "pink",
    "0.3": "blue",
    "1.0":"green",
    "3.0": "red",
    
    "12.0": "red",
    "24.0": "green",
    
    "1": "red",
    "5": "green",
    "10": "blue",
    "100": "brown",
    "1000": "black"
}

# CLASS_CMAP = np.array(["tab:blue","tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", :"tab:cyan"])

MARKERS = {
    "fmn": "o",
    "alma": "v",
    "apgd": "s",
    "ensemble": "*",
    "pdgd": "+",
    "pdpgd": "+",
    "ddn": "2",
    "aa": 3,
    "auto_all": 3
}

STYLE = lambda steps: "-" if steps >= 1000 else ":"



class Logger:
    def __init__(self, logs_dir="logs"):
        self.dict = defaultdict(dict) #defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(None))))
        self.logs_dir = logs_dir
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
            return run_id.split(f"-{parameter_name}-")[1].split("-")[0]
        except:
            try:
                return run_id.split(f"{parameter_name}-")[1].split("-")[0]
            except:
                pass
        
    def row_from_id(self, run_id, index):
        row = dict()
        for par in index[:-2]:
            row[par] = self.value_from_id(par, run_id)
            
        row["steps"] = self.get_n_steps(run_id)
        row["params"] = run_id.split(row["model"])[1]
        return pd.Series(row, index=index)
    
    def where(self, **kwargs):
        new = dict()
        
        for key in self.dict:
            if not all((param not in key or f"{param}-{kwargs[param]}" in key for param in kwargs)):
                continue
            new[key] = self.dict[key]
        return new


    def save(self, run_id=None, force=False):
        self.single_log("end_time", time.time())
        
        # use default
        if run_id is None:
            run_id = self.run_id

        path = f"{self.logs_dir}/{run_id}"

        if not force and os.path.exists(path):
            raise RuntimeWarning(run_id)

        # make dir
        os.makedirs(path, exist_ok=True)

        for attr in self.dict[run_id]:
            np.save(f"{self.logs_dir}/{run_id}/{attr}.npy", self.dict[run_id][attr])
            
    def save_all(self, force=False):

        for run_id in self.dict:
            try:
                self.save(run_id, force)
            except RuntimeWarning as e:
                run_id = e.args
                print(f"Log of {run_id} already saved ... skipping")

    def load(self, run_id, force=False):

        if not force and run_id in self.dict:
            raise RuntimeWarning("Record already loaded in dict")

        for entry in os.scandir(f"{self.logs_dir}/{run_id}"):
            self.dict[run_id][entry.name[:-4]] = np.load(f"{self.logs_dir}/{run_id}/{entry.name}",allow_pickle=True)

    def load_all(self, force=False):
        for run_id in os.listdir(self.logs_dir):
            if not os.path.isdir(os.path.join(self.logs_dir, run_id)):
                continue
            try:
                self.load(run_id, force=force)
            except RuntimeWarning as e:
                print(f"{run_id} already loaded.")

    def is_logged(self, run_id):
        path = os.path.join(self.logs_dir, run_id)
        return os.path.exists(path)

    
    def get_n_steps(self, run_id):
        # get the real number of forwards
        if run_id in self.dict:
            return len(self.dict[run_id]["norm_progress"])

        raise NotImplementedError("Not ready for not runned experiments")
        # # estimate
        # attack_name = self.value_from_id("attack", run_id)
        # if attack_name == "fmn":
        #     return self.value_from_id("steps", run_id)
        # if attack_name == "alma":
        #     return self.value_from_id("steps", run_id)
    
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
        
        columns = ["dataset", "norm", "model", "attack", "steps", "params", "median", "n_qs", "asr", "first_adv_step", "first_last_ratio"]
        df = pd.DataFrame(columns=columns)
        
        for run_id in dic:
            step_norms = dic[run_id]["norm_progress"]
            step_accs = dic[run_id]["acc_progress"]
            qd = QD(step_norms, step_accs)
            n_qs = n_qs_to_reach(0.1,qd)
            mm = min_median(qd)
            asr = attack_succes_rate(step_accs)
            
            first_adv_step = np.nanmean(steps_to_find_adv(step_accs))
            first_last_ratio = np.nanmean(first_to_final_ratio(step_norms, step_accs))
            
            s = pd.concat([self.row_from_id(run_id, index=columns[:-5]), (pd.Series((mm, n_qs, asr, first_adv_step, first_last_ratio)))])
            s.name = run_id
            s.index= columns
            df = df.append(s)
        return df    

    def report_nqs(self, dic=None):
        if dic is None:
            dic = self.dict
        
        columns = ["dataset", "norm", "model", "attack", "steps", "params", "n_qs"]
        df = pd.DataFrame(columns=columns)
        
        for run_id in dic:
            n_qs = n_qs_to_reach(0.1, dic[run_id]["norm_progress"], dic[run_id]["acc_progress"])
            df = df.append(self.row_from_id(run_id, index=columns[:-1]).
                           append(pd.Series(mm, index=[columns[-1]])), ignore_index=True)
        return df    
        
    def report_medians(self, dic=None):
        if dic is None:
            dic = self.dict
        
        columns = ["dataset", "norm", "model", "attack", "steps", "params", "median"]
        df = pd.DataFrame(columns=columns)
        
        for run_id in dic:
            mm = min_median(dic[run_id]["norm_progress"], dic[run_id]["acc_progress"])
            df = df.append(self.row_from_id(run_id, index=columns[:-1]).
                           append(pd.Series(mm, index=[columns[-1]])), ignore_index=True)
            # print(df, self.row_from_id(run_id))
            # df.iloc[-1]["median"] = mm
        return df   
    
    def report_ra(self, dic, eps=8/255):
        columns = ["dataset", "norm", "model", "attack", "steps", "params", "rob_acc"]
        df = pd.DataFrame(columns=columns)
        
        for run_id in dic:
            step_norms = dic[run_id]["norm_progress"]
            step_accs = dic[run_id]["acc_progress"]
            ra, epss = SE(step_norms, step_accs, n_thrs=1, fixed_thrs=[eps], max_eps=0)
            df = df.append(self.row_from_id(run_id, index=columns[:-1]).
                           append(pd.Series(ra[epss==eps], index=[columns[-1]])), ignore_index=True)
        return df
    
    def plot_SE(self, run_id=None, ax=None, compare_on="attack", max_eps="auto"):
        # use default
        if run_id is None:
            run_id = self.run_id
        
        norm_name = self.value_from_id("norm", run_id)
        model_name = self.value_from_id("model", run_id)
        fixed_thrs = THRESHOLDS[norm_name]
        reported_scores = SCORES[norm_name]
        steps = self.get_n_steps(run_id)
        
        if ax is None:
            ax = plt.gca()
        name = self.value_from_id(compare_on, run_id)
        if name is None:
            return
        
        
        ax.set_ylim(bottom=0)

        # plot reported scores 
        for i, t in enumerate(fixed_thrs):
            # plot fixed threshold
            ax.axvline(x = t, color = 'grey')
            
            # plot reported scores on robustbench
            if model_name in reported_scores[i]:
                ax.scatter(t, reported_scores[i][model_name], marker="o", color="black")
#                 ax.axhline(reported_scores[i][model_name], color="grey")
        
        # plot points if aa
        if self.value_from_id("attack", run_id) == "aa":
            acc = self.dict[run_id]["acc_progress"][-1].mean()
            ax.scatter(float(self.value_from_id("eps", run_id)), acc, marker="o", label="aa", color=COLORS["aa"])
            return 
                
        
        if max_eps == "auto":
            if norm_name == "L1":
                max_eps = fixed_thrs[-1]*3
            elif norm_name == "L2":
                max_eps = fixed_thrs[-1]*4
            elif norm_name == "Linf":
                max_eps = fixed_thrs[-1]*6
                
        # plot normal minimal norm attack
        rob_accs, eps_thr = SE(self.dict[run_id]["norm_progress"], self.dict[run_id]["acc_progress"], fixed_thrs=fixed_thrs, max_eps=max_eps)
        ax.plot(eps_thr, rob_accs, label=name, color=COLORS[name], linestyle=STYLE(steps))
        
            
        ax.legend(loc='upper right')
    
    def plot_QD(self, run_id=None, ax=None, compare_on="attack"):
        # use default
        if run_id is None:
            run_id = self.run_id

        qd = QD(self.dict[run_id]["norm_progress"], self.dict[run_id]["acc_progress"])

        if ax is None:
            ax = plt.gca()
        name = self.value_from_id(compare_on, run_id)
        if name is None:
            return
        
        steps = self.get_n_steps(run_id)

        ax.plot(qd, label=name, color=COLORS[name], linestyle=STYLE(steps))
        ax.legend(loc='upper left')
        ax.set_xscale("log")
   
    def plot_targeted_class(self, run_id, ax=None, attack="fmn", sumdic=None, selected=None, *args, **kwargs):
#         if self.value_from_id("attack", run_id) != attack:
#             return
        # only long runs
        if self.get_n_steps(run_id) < 1000:
            return
        
        # only one plot per ax
        assert not ax.collections
        
        all_window = []
        all_init = []
        for i, sample in enumerate(range(len(self.dict[run_id]["acc_progress"][0]))):
#             tar = [(x[0], x[1], self.dict[run_id]["acc_progress"][j][i]) for j, x in enumerate(self.dict[run_id]["pred_progress"][:, i].argsort(axis=1)[:,::-1])]
#             print(tar)
            tar = [x[int(self.dict[run_id]["acc_progress"][j][i])] for j, x in enumerate(self.dict[run_id]["pred_progress"][:, i].argsort(axis=1)[:,::-1])]
            
            tar = np.array(tar)
            window_size = 5
            tar_window1 = []
            for j in range(0, len(tar), window_size):
                unique, counts = np.unique(tar[j:j+window_size], return_counts=True) 
                
                argsorted = np.argsort(counts)[::-1]
#                 print(unique)
                tar_window1.append(unique[argsorted[0]])
#                 tar_window2.append(unique[argsorted[min(1, len(unique)-1)]])
#             print(tar_window1)
            all_window.append(tar_window1)
#             print(tar[0])
            all_init.append(-1 if self.dict[run_id]["acc_progress"][0][i] == 0 else tar[0])
#             print(tar[0], tar_window1[0])
        def argsort(seq):
            return sorted(range(len(seq)), key=seq.__getitem__)
        
        argsorted = argsort(all_window)
        
        # keep only selected indeces
        if selected is not None:
            argsorted = [x for x in argsorted if selected[self.value_from_id("norm", run_id)][self.value_from_id("model", run_id)][x]]

        all_window = np.array(all_window)[argsorted]
        all_init = np.array(all_init)[argsorted]
        y=np.repeat(np.arange(len(all_window))/30, len(all_window[0]))
        
        # initially targeted class
        ax.scatter(x=np.repeat(-5, len(all_init)), y=np.arange(len(all_init))/30, c=all_init, s=(all_init != -1).astype(int)*50,
                   cmap="tab10", marker="+", norm=NORM10)
        
        # attack 
        ax.scatter(x=np.tile(np.arange(len(all_window[0])), len(all_window)),y=y, c=all_window, cmap="tab10", norm=NORM10)
#             ax.scatter(x=np.arange(len(tar_window2)),y=np.ones(len(tar_window2))*i-0.02, c=tar_window2)
#             if i > 50:
#                 break

        # final class which best-evaded the classifier
        min_norms, min_preds = min_norms_pred(self.dict[run_id]["norm_progress"], \
                                                          self.dict[run_id]["acc_progress"], \
                                                          self.dict[run_id]["pred_progress"])
        all_attack_best = min_preds.argmax(axis=-1)[argsorted]
        ax.scatter(x=np.repeat(len(all_window[0])+5, len(all_init)), y=np.arange(len(all_init))/30, c=all_attack_best, s=(all_init != -1).astype(int)*50,
                   cmap="tab10", marker="+", norm=NORM10)
        
        # optionally add optimal class from sumdic
        if sumdic is not None:
            curr = sumdic[self.value_from_id("norm", run_id)][self.value_from_id("model", run_id)]
            curr_i = curr["run_ids"].index(run_id)
            all_best = np.take_along_axis(curr["classes"], curr["norms"].argmin(axis=0, keepdims=True), axis=0)[0][argsorted]
                
            ax.scatter(x=np.repeat(len(all_window[0])+10, len(all_init)), y=np.arange(len(all_init))/30, c=all_best, s=(all_init != -1).astype(int)*50,
                       cmap="tab10", marker="*", norm=NORM10)
            
#             ax.scatter(x=np.repeat(len(all_window[0])+10, len(all_init)), y=np.arange(len(all_init))/30, c=black, s=(all_init != -1).astype(int)*50,
#                        cmap="tab10", marker="*", norm=NORM10)

            n = (min_norms/(curr["norms"].min(axis=0)))[argsorted]
#             print(curr["norms"].min(axis=0), min_norms)
            for i, txt in enumerate(n):
                txt = f"{txt:.3}"
                ax.annotate(txt, ((np.repeat(len(all_window[0])+10, len(all_init))[i], (np.arange(len(all_init))/30)[i])))
            
    def plot_QD_grid(self,*args, **kwargs):
        self.plot_grid(plot_fun=self.plot_QD, *args, **kwargs)

    def plot_SE_grid(self,*args, **kwargs):
        self.plot_grid(plot_fun=self.plot_SE, *args, sharex="none",**kwargs)

        
    def plot_grid(self, plot_fun, where_settings=dict(), x_axis="model", y_axis="norm", aggregate="best", 
                  compare_on="attack", sharex="row", max_count_x=10, **kwargs):
        dic = self.where(**where_settings)
        comp = list(dic.keys())
        
        unique_y = list(sorted(set([self.value_from_id(y_axis, c) for c in comp])))
        unique_x_per_y = [list(sorted(set([self.value_from_id(x_axis, c) for c in filter(lambda x: y in x, dic.keys())]))) for y in unique_y]
        
        # sort according to scores on the first threshold
        for i, row in enumerate(unique_x_per_y):
            y = unique_y[i]
            row.sort(key=lambda x: SCORES[y][0][x] if x in SCORES[y][0] else -1, reverse=True)
        
        # reduce the count of displayed plots to the limit
        if max_count_x is not None:
            unique_x_per_y = [row[:max_count_x] for row in unique_x_per_y]
            
        if aggregate == "best":
            # TODO select best hyperparams for each x,y pair
            for c in comp:
                pass
        
        fig, axs = plt.subplots(len(unique_y), max(map(len, unique_x_per_y)), sharey='none', sharex=sharex, **kwargs)
        
        # at least 2D
        if len(unique_y) == 1:
            axs = [axs]
        
        for c in comp:
            x_val = self.value_from_id(x_axis, c)
            y_val = self.value_from_id(y_axis, c)
            
            y_index = unique_y.index(y_val)
            
            # skip the models with rank below plot count limit
            if x_val not in unique_x_per_y[y_index]:
                continue
            
            
            ax = axs[y_index][unique_x_per_y[y_index].index(x_val)]
            plot_fun(c, ax=ax, compare_on=compare_on)
            
            ax.set_title(x_val)
        plt.show()
        
    def merge(self, name, run_ids, strategy="best"):
        if strategy != "best":
            raise NotImplementedError
            
        # get longest run
        n_steps = [self.get_n_steps(run_id) for run_id in run_ids]
        longest_ind = n_steps.index(max(n_steps))
        
        dic = deepcopy(self.dict[run_ids[longest_ind]])
        for run_id in run_ids:
            clean_acc = dic["acc_progress"][0]
            max_ind = len(self.dict[run_id]["norm_progress"])-1
#             print(run_id, run_ids[0])
#             print(clean_acc == self.dict[run_id]["acc_progress"][0])
            # all clean accuracies should be the same
            assert (self.dict[run_ids[0]]["labels_progress"] == self.dict[run_id]["labels_progress"]).all()
#             assert (self.dict[run_ids[0]]["pred_progress"] == self.dict[run_id]["pred_progress"]).all()
#             assert (self.dict[run_ids[0]]["acc_progress"][0] == self.dict[run_id]["acc_progress"][0]).all()
            

            if not (clean_acc == self.dict[run_id]["acc_progress"][0]).all():
                raise RuntimeError(run_id)
                        
            for step in range(len(dic["norm_progress"])):
                step_norm1 = adversify(dic["norm_progress"][step], dic["acc_progress"][step], clean_acc)
                step_norm2 = adversify(self.dict[run_id]["norm_progress"][min(step, max_ind)], self.dict[run_id]["acc_progress"][min(step, max_ind)], clean_acc)
            
                dic["norm_progress"][step] = np.amin(np.stack([step_norm1, step_norm2]), axis=0)
                dic["acc_progress"][step] = np.amin(np.stack([dic["acc_progress"][step], self.dict[run_id]["acc_progress"][min(step, max_ind)]]), axis=0)
        
        self.dict[name] = dic
        return dic
        
    def ensemble(self, OPTIMAL_HYPERS, SCORES):
        for norm in OPTIMAL_HYPERS:
            for model_name in SCORES[norm][0]:
                run_ids = [*filter( lambda x: "attack-aa" not in x and "auto_all" not in x, list(self.where(model=model_name, **OPTIMAL_HYPERS[norm]).keys()))]

                is_slow = lambda x: self.get_n_steps(x) > 999
                quick_ids = [*filter(lambda x: not is_slow(x), run_ids)]
                slow_ids = [*filter(is_slow, run_ids)]

                self.merge(f"dataset-CIFAR10-norm-{norm}-attack-ensemble-model-{model_name}-steps-1001", slow_ids)
                self.merge(f"dataset-CIFAR10-norm-{norm}-attack-ensemble-model-{model_name}-steps-101", quick_ids)
                
    def extract_sumdic(self, OPTIMAL_HYPERS, SCORES):
        res_s = {key:dict() for key in OPTIMAL_HYPERS}
        res_q = {key:dict() for key in OPTIMAL_HYPERS}
        for norm in OPTIMAL_HYPERS:
            for model_name in SCORES[norm][0]:
                res_s[norm][model_name] = dict()
                res_q[norm][model_name] = dict()
                
                run_ids = [*filter( lambda x: "" in x, list(self.where(model=model_name, **OPTIMAL_HYPERS[norm]).keys()))]

                is_slow = lambda x: self.get_n_steps(x) > 999 or self.value_from_id("attack", x) == "auto_all" or self.value_from_id("attack", x) == "aa"
                quick_ids = [*filter(lambda x: not is_slow(x) or self.value_from_id("attack", x) == "auto_all", run_ids)]
                slow_ids = [*filter(is_slow, run_ids)]
                for res,ids in zip([res_s, res_q],[slow_ids, quick_ids]):

                    norms = np.zeros(shape=(len(ids),len(self.dict[ids[0]]["labels_progress"][0])))
                    classes = np.zeros_like(norms, dtype=int)

                    for i, run_id in enumerate(ids):
                        min_norms, min_preds = min_norms_pred(self.dict[run_id]["norm_progress"], \
                                                              self.dict[run_id]["acc_progress"], \
                                                              self.dict[run_id]["pred_progress"])
                        norms[i] = min_norms
                        classes[i] = min_preds.argmax(axis=-1)

                    res[norm][model_name]["norms"] = norms
                    res[norm][model_name]["classes"] = classes
                    res[norm][model_name]["run_ids"] = ids
                    res[norm][model_name]["clean_preds"] = self.dict[ids[0]]["pred_progress"][0]
        
        return res_s, res_q
    
    def plot_comparison(self, norm_a, norm_b, classes_a, classes_b, run_id_a, run_id_b, ax, size_f=None, simple=False):
        if size_f is None:
            size = lambda classes_a, classes_b: [1 if classes_a[i] == classes_b[i] else 500 for i in range(len(classes_a))]
        else:
            size = size_f
        
        
        
        if simple:
            cs =  ["green" if classes_a[i] == classes_b[i] else "red" for i in range(len(classes_a))]
            scatter = ax.scatter(x=norm_a, y=norm_b ,c=cs, s=50, marker="o", 
                             cmap="tab10", norm=NORM10)
            
            uni, counts = np.unique(cs, return_counts=True)
            for i, c in enumerate(uni):
                ax.scatter([], [], c=c, s=counts[i],label=str(counts[i]))
        
            ax.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Proportions') 
        
        else:
            scatter = ax.scatter(x=norm_a, y=norm_b ,c=classes_a, s=size(classes_a, classes_b), edgecolor="black", marker=MarkerStyle("o", fillstyle="right"), 
                                 cmap="tab10", norm=NORM10)
            ax.scatter(x=norm_a, y=norm_b ,c=classes_b, s=size(classes_a, classes_b), edgecolor="black", marker=MarkerStyle("o", fillstyle="left"), 
                       cmap="tab10", norm=NORM10)
        
            # produce a legend with the unique colors from the scatter
            legend1 = ax.legend(*scatter.legend_elements(),
                                loc="lower right", title="Classes")
            ax.add_artist(legend1)

        ax.set_ylabel(run_id_b)
        ax.set_xlabel(run_id_a)
        al = max(ax.get_ylim()[1], ax.get_xlim()[1])
        alm = min(ax.get_ylim()[0], ax.get_xlim()[0])
        ax.set_ylim(bottom= alm, top=al)
        ax.set_xlim(left = alm, right=al)
        
        return norm_a > norm_b#np.nonzero(norm_a > norm_b)
        
    def plot_compare_grid(self, sumdic, a="fmn", b="apgd", x_axis="model", y_axis="norm", aggregate="best",\
                          sharex='none', max_count_x=10, size_f=None, simple=None, **kwargs):
        
        unique_y = list(sorted(sumdic.keys()))
        unique_x_per_y = [list(sorted([*sumdic[y].keys()])) for y in unique_y]
        
        # sort according to scores on the first threshold
        for i, row in enumerate(unique_x_per_y):
            y = unique_y[i]
            row.sort(key=lambda x: SCORES[y][0][x] if x in SCORES[y][0] else -1, reverse=True)
        
        # reduce the count of displayed plots to the limit
        if max_count_x is not None:
            unique_x_per_y = [row[:max_count_x] for row in unique_x_per_y]
            
#         if aggregate == "best":
#             # TODO select best hyperparams for each x,y pair
#             for c in comp:
#                 pass
        
        fig, axs = plt.subplots(len(unique_y), max(map(len, unique_x_per_y)), sharey='none', sharex=sharex, **kwargs)
        
        # at least 2D
        if len(unique_y) == 1:
            axs = [axs]
                
        # gather critical indices
        critical = dict()
        
        for i, y_val in enumerate(unique_y):
            critical[y_val] = dict()
            for j, x_val in enumerate(unique_x_per_y[i]):
                            
                ax = axs[i][j]

                a_indcs = [i for i,x in enumerate(sumdic[y_val][x_val]["run_ids"]) if self.value_from_id("attack", x) == a]
                b_indcs = [i for i,x in enumerate(sumdic[y_val][x_val]["run_ids"]) if self.value_from_id("attack", x) == b]
                
                if not a_indcs or not b_indcs:
                    continue
                
                assert len(a_indcs) == 1 and len(b_indcs) == 1
            
                critical[y_val][x_val] = self.plot_comparison(sumdic[y_val][x_val]["norms"][a_indcs[-1]], 
                                     sumdic[y_val][x_val]["norms"][b_indcs[-1]], 
                                     sumdic[y_val][x_val]["classes"][a_indcs[-1]], 
                                     sumdic[y_val][x_val]["classes"][b_indcs[-1]], 
                                     sumdic[y_val][x_val]["run_ids"][a_indcs[-1]], 
                                     sumdic[y_val][x_val]["run_ids"][b_indcs[-1]],
                                     ax, 
                                     size_f=size_f,
                                    simple=simple)
                ax.set_title(x_val)
                    
                # plot help lines
                ax.plot([0, 1], [0, 1], transform=ax.transAxes, linewidth=0.3)
                if b == "auto_all":
                    ax.axvline(x = THRESHOLDS[y_val][0], color = 'grey', linewidth=0.3)
                elif a == "auto_all":
                    ax.axhline(y = THRESHOLDS[y_val][0], color = 'grey', linewidth=0.3)
        plt.show()    
        return critical
        
#         for c in comp:
#             x_val = self.value_from_id(x_axis, c)
#             y_val = self.value_from_id(y_axis, c)
            
#             y_index = unique_y.index(y_val)
            
#             # skip the models with rank below plot count limit
#             if x_val not in unique_x_per_y[y_index]:
#                 continue
            
            
#             ax = axs[y_index][unique_x_per_y[y_index].index(x_val)]
#             plot_fun(c, ax=ax, compare_on=compare_on)
            
#             ax.set_title(x_val)
#         plt.show()    


    def plot_comparison_all(self, dic, ax, selected=None, initial=None):
#         if size_f is None:
#             size = lambda classes_a, classes_b: [1 if classes_a[i] == classes_b[i] else 500 for i in range(len(classes_a))]
#         else:
#             size = size_f
        if selected == "skip_same":
            mask = dic["classes"].max(axis=0) != dic["classes"].min(axis=0)
        elif selected == "only_same":
            mask = dic["classes"].max(axis=0) == dic["classes"].min(axis=0)
        elif selected is None: 
            mask = np.zeros_like(dic["classes"][0]) == 0
        else:
            mask = selected
        norms =dic["norms"][:,mask]
        classes =dic["classes"][:,mask]
        second_clean = np.argsort(dic["clean_preds"][mask,:], axis=1)[:,-2]
        argsorted = np.argsort(norms.min(axis=0))
#         print(argsorted)
#         print(dic["norms"].mean(axis=0)[argsorted])
            
        # add second initial class
        if initial in ["follow", "fix"]:
            y = np.ones_like(argsorted, dtype=float)
            if initial == "fix":
                y*=-.2 
            elif initial == "follow":
                y = norms.min(axis=0)[argsorted] - norms.min(axis=0)[argsorted].max() * 0.01
            scatter = ax.scatter(x=np.arange(len(argsorted)), y=y, c=second_clean, \
                                      marker="x", cmap="tab10", norm=NORM10)

    
        for i, run_id in enumerate(dic["run_ids"]):
            if self.value_from_id("attack",run_id) == "ensemble":
                continue
                
            scatter = ax.scatter(x=np.arange(len(argsorted)), y=norms[i][argsorted], c=classes[i], \
                                  marker=MARKERS[self.value_from_id("attack",run_id)], cmap="tab10", 
                                 label=self.value_from_id("attack",run_id), norm=NORM10)
#         ax.scatter(x=norm_a, y=norm_b ,c=classes_b, s=size(classes_a, classes_b), edgecolor="black", marker=i, cmap="tab10")
        # produce a legend with the unique colors from the scatter
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="lower right", title="Classes")
        
        ax.add_artist(legend1)
#         ax.set_ylabel(run_id_b)
#         ax.set_xlabel(run_id_a)
#         al = max(ax.get_ylim(), ax.get_xlim())
#         ax.set_ylim(al)
#         ax.set_xlim(al)
        ax.legend()

    
        
    
    def plot_compare_all_grid(self, sumdic, x_axis="model", y_axis="norm", aggregate="best",\
                          sharex='none', max_count_x=10, initial=None, selected=None, **kwargs):
        
        unique_y = list(sorted(sumdic.keys()))
        unique_x_per_y = [list(sorted([*sumdic[y].keys()])) for y in unique_y]
        
        # sort according to scores on the first threshold
        for i, row in enumerate(unique_x_per_y):
            y = unique_y[i]
            row.sort(key=lambda x: SCORES[y][0][x] if x in SCORES[y][0] else -1, reverse=True)
        
        # reduce the count of displayed plots to the limit
        if max_count_x is not None:
            unique_x_per_y = [row[:max_count_x] for row in unique_x_per_y]
            
#         if aggregate == "best":
#             # TODO select best hyperparams for each x,y pair
#             for c in comp:
#                 pass
        
        fig, axs = plt.subplots(len(unique_y), max(map(len, unique_x_per_y)), sharey='none', sharex=sharex, **kwargs)
        
        # at least 2D
        if len(unique_y) == 1:
            axs = [axs]
                
        
        for i, y_val in enumerate(unique_y):
            for j, x_val in enumerate(unique_x_per_y[i]):
                            
                ax = axs[i][j]

#                 a_indcs = [i for i,x in enumerate(sumdic[y_val][x_val]["run_ids"]) if self.value_from_id("attack", x) == a]
#                 b_indcs = [i for i,x in enumerate(sumdic[y_val][x_val]["run_ids"]) if self.value_from_id("attack", x) == b]
                
#                 if not a_indcs or not b_indcs:
#                     continue
#                 assert len(a_indcs) == 1 and len(b_indcs) == 1
                sel = selected[y_val][x_val] if isinstance(selected, dict) else selected
                self.plot_comparison_all(sumdic[y_val][x_val], ax, initial=initial, selected=sel)
                ax.set_title(x_val)
    
        plt.show()    

        
    def select_add_best_wrt_model(self):
        pass
    