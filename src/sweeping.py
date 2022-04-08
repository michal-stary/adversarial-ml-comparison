import json
import pandas as pd
from sklearn.model_selection._search import ParameterGrid
from logging_ import Logger
from adv_lib.attacks import fmn, alma
from robustbench.utils import load_model
from tracking import PyTorchModelTrackerSetup
from torchvision import transforms
import torch
from src.utils.data_utils import create_loaders
from tqdm import tqdm



class Sweeper:
    @classmethod
    def from_jsonfile(cls, path, *args, **kwargs):
        with open(path, "r") as f_json:
            dic = json.load(f_json)
        df = Sweeper.build_df_from_dict(dic)
        return Sweeper(df, *args, **kwargs)

    @classmethod
    def build_df_from_dict(cls, d):
        # make pandas df from the json config (cartesian product)
        g = ParameterGrid(d)
        # print([*g])
        df = pd.DataFrame([*g]).drop_duplicates().convert_dtypes()
        cols_to_order = ['dataset', 'attack', "norm", "model"]
        new_columns = cols_to_order + (df.columns.drop(cols_to_order).tolist())
        return df[new_columns].sort_values(cols_to_order).reset_index(drop=True)

    @classmethod
    def from_csvfile(cls, path, *args, **kwargs):
        df = pd.read_csv(path)

        return Sweeper(df, *args, **kwargs)

    def __init__(self, df, log_dir="logs", model_dir="models", data_dir="data"):
        self.config_df = df
        self.logger = Logger()
        self.model_dir = model_dir
        self.data_dir = data_dir


    def sweep(self, n_samples=100, recompute=False, logs_dir="logs"):
        for index, row in self.config_df.iterrows():
            row = row.dropna()

            # create id
            run_id = "-".join(row.dropna().to_string(header=False).split())

            if not recompute and self.logger.is_logged(run_id, logs_dir):
                try:
                    self.logger.load(run_id, dir=logs_dir)
                except Warning:
                    pass
                continue
            print(row)

            # run attack
            self.logger.setup(run_id=run_id)
            # self.logger.setup(attack_name=row.attack, model_name=row.model, hyperparams="default")
            self.run(n_samples=n_samples, **row.to_dict())

            # save attack log to disk
            self.logger.save(force=True)

            # print progress
            print(f"Done: {index+1}/{self.config_df.shape[0]}")

    def run(self, n_samples=100, batch_size=20, **kwargs):
        if "attack" in kwargs:
            if kwargs["attack"] == "fmn":
                attack_f = fmn
            elif kwargs["attack"] == "alma":
                attack_f = alma
        else:
            raise Warning()

        if "norm" in kwargs:
            if attack_f == fmn:
                numerical_part = kwargs["norm"][1:]
                numeric_norm = float(numerical_part)
                norm_dict = {"norm":numeric_norm}
            elif attack_f == alma:
                norm_dict = {"distance": kwargs["norm"].lower()}


        if "model" in kwargs and "dataset" in kwargs and "norm" in kwargs:
            model = load_model(model_name=kwargs["model"], dataset=kwargs["dataset"].lower(),
                               threat_model=kwargs["norm"], model_dir=self.model_dir)
        else:
            raise Warning()

        tracked_model = PyTorchModelTrackerSetup(model, loss_f=kwargs["loss_f"], logger=self.logger)  # pytorch model

        hyperparams = {**kwargs}
        hyperparams.pop("model")
        hyperparams.pop("dataset")
        hyperparams.pop("attack")
        hyperparams.pop("norm")
        hyperparams.pop("loss_f")

        n_workers = 1
        data_loader = create_loaders(self.data_dir, task_config=kwargs["dataset"],
                                     batch_size=batch_size,
                                     transform=None,
                                     random_state=0,
                                     n_workers=n_workers)

        for b, (images, labels) in enumerate(tqdm(data_loader, total=n_samples // batch_size)):
            if b * batch_size >= n_samples:
                break

            tracked_model.setup(images, labels)

            # run attack
            results = attack_f(tracked_model, images, labels, **norm_dict, **hyperparams)
            tracked_model.log()


    def save_df(self, path="logs/logmap.csv"):
        self.config_df.to_csv(path)


