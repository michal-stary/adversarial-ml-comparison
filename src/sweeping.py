import json
import pandas as pd
from pandas import json_normalize
from flatten_json import flatten
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
    def from_jsonfile(cls, path):
        with open(path, "r") as f_json:
            dic = json.load(f_json)
        df = Sweeper.build_df_from_dict(dic)
        return Sweeper(df)

    @classmethod
    def build_df_from_dict(cls, d):
        # make pandas df from the json config (cartesian product)
        g = ParameterGrid(d)
        # print([*g])
        return pd.DataFrame([*g]).drop_duplicates().convert_dtypes()

    @classmethod
    def from_csvfile(cls, path):
        df = pd.read_csv(path)

        return Sweeper(df)

    def __init__(self, df, log_dir="logs", model_dir="models", data_dir="data"):
        self.config_df = df
        self.logger = Logger()
        self.model_dir = model_dir
        self.data_dir = data_dir


    def sweep(self, n_samples=100, recompute=False, logs_dir="logs"):
        for index, row in self.config_df.iterrows():
            row = row.dropna()

            # TODO - check if row not already logged
            if not recompute and self.logger.is_logged(index, logs_dir):
                continue
            print(row)
            # run attack
            self.logger.setup(run_id=index)
            # self.logger.setup(attack_name=row.attack, model_name=row.model, hyperparams="default")
            self.run(n_samples=n_samples, **row.to_dict())

            # save attack log to disk
            self.logger.save(force=True)

            # print progress
            print(f"Done: {index}/{self.config_df.shape[0]}")

    def run(self, n_samples=100, batch_size=20, **kwargs):
        if "attack" in kwargs:
            if kwargs["attack"] == "fmn":
                attack_f = fmn
            elif kwargs["attack"] == "alma":
                attack_f = alma
        else:
            raise Warning()

        if "norm" in kwargs:
            numerical_part = kwargs["norm"][1:]
            numeric_norm = float(numerical_part)

        if "model" in kwargs and "dataset" in kwargs and "norm" in kwargs:
            model = load_model(model_name=kwargs["model"], dataset=kwargs["dataset"].lower(),
                               threat_model=kwargs["norm"], model_dir=self.model_dir)
        else:
            raise Warning()

        tracked_model = PyTorchModelTrackerSetup(model, loss_f="CE", logger=self.logger)  # pytorch model

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
            results = attack_f(tracked_model, images, labels, numeric_norm, **hyperparams)
            tracked_model.log()


    def save_df(self, path="logmap.csv"):
        self.config_df.to_csv(path)


