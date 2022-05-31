import json
import pandas as pd
from sklearn.model_selection._search import ParameterGrid
from logging_ import Logger
from adv_lib.attacks import fmn, alma, apgd, ddn, pdgd, pdpgd
from adv_lib.attacks.auto_pgd import minimal_apgd
#from robustbench.utils import load_model
from tracking import PyTorchModelTracker
from torchvision import transforms
import torch
from utils.data_utils import create_loaders
from tqdm import tqdm
from zoo import load_model
from autoattack_wrapper import aa

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
        cols_to_order = ['dataset', "norm","attack", "model"]
        new_columns = cols_to_order + (df.columns.drop(cols_to_order).tolist())
        return df[new_columns].sort_values(cols_to_order).reset_index(drop=True)

    @classmethod
    def from_csvfile(cls, path, *args, **kwargs):
        df = pd.read_csv(path)

        return Sweeper(df, *args, **kwargs)

    def __init__(self, df, log_dir="logs", model_dir="models", data_dir="data"):
        self.config_df = df
        self.logger = Logger(logs_dir=log_dir)
        self.model_dir = model_dir
        self.data_dir = data_dir


    def sweep(self, n_samples=100, recompute=False, device=None, batch_size=None, n_workers=0):
        for index, row in self.config_df.iterrows():
            row = row.dropna()

         
            def id_from_row(row):
                # create id   
                pre = "-".join(row.iloc[:4].to_string(header=False).split()) 
                
                rest = "-".join(row.iloc[4:].sort_index().to_string(header=False).split())
                return pre + "-" + rest
            
            
            run_id = id_from_row(row)
            print(run_id)
            # load from precomputed if possible
            if not recompute and self.logger.is_logged(run_id):
                try:
                    self.logger.load(run_id)
                except Warning:
                    pass
                continue
            print(row)

            # get device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            try:
                # run attack
                self.logger.setup(run_id=run_id, n_samples=n_samples)
                self.run(n_samples=n_samples, batch_size=batch_size, device=device, n_workers=n_workers,**row.to_dict())

                # save attack log to disk
                self.logger.save(force=True)
            
            except KeyError as e:
                print(e)
                print(f"{run_id} not available.")
                
            # print progress
            print(f"Done: {index+1}/{self.config_df.shape[0]}")

    def run(self, n_samples=100, batch_size=20, n_workers=0, device="cpu", **kwargs):
        if "attack" in kwargs:
            if kwargs["attack"] in ["fmn", "fmn_t"]:
                attack_f = fmn
            elif kwargs["attack"] in ["alma", "alma_t"]:
                attack_f = alma
            elif kwargs["attack"] == "apgd":
                attack_f = minimal_apgd
            elif kwargs["attack"] == "ddn":
                attack_f = ddn
            elif kwargs["attack"] == "pdpgd":
                attack_f = pdpgd
            elif kwargs["attack"] == "pdgd":
                attack_f = pdgd
            elif kwargs["attack"] == "aa":
                attack_f = aa
        else:
            raise Warning()


        # parse norm
        if "norm" in kwargs:
            numerical_part = kwargs["norm"][1:]
            numeric_norm = float(numerical_part)
            if attack_f in [fmn, minimal_apgd, pdpgd]:
                norm_dict = {"norm":numeric_norm}
            elif attack_f == alma:
                norm_dict = {"distance": kwargs["norm"].lower()}
            elif attack_f in [ddn, pdgd]:
                norm_dict = {}
            elif attack_f == aa:
                norm_dict = {"norm":kwargs["norm"]}
        else:
            raise RuntimeError(f"Norm not present in {kwargs}")
        
        
        
        if "init_aa_eps" in kwargs:
            aa_run_id = f'dataset-{kwargs["dataset"]}-norm-{kwargs["norm"]}-attack-aa-model-{kwargs["model"]}-eps-{round(kwargs["init_aa_eps"], 6)}-loss_f-DL'
#             print(self.logger.dict[aa_run_id])
            print(aa_run_id)
            initial = torch.tensor(self.logger.dict[aa_run_id]["results"][0]).to(device)
#             print(initial.shape)
            kwargs["starting_points"] = initial        
            kwargs.pop("init_aa_eps")
            
        # load model
        if "model" in kwargs and "dataset" in kwargs and "norm" in kwargs:
            model = load_model(model_name=kwargs["model"], dataset=kwargs["dataset"].lower(),
                               model_dir=self.model_dir)

        #
        model.eval()
        model = model.to(device)


        tracked_model = PyTorchModelTracker(model, p=numeric_norm, logger=self.logger, loss_f=kwargs["loss_f"], track_acc=True)  # pytorch model

        # tracked_model = PyTorchModelTracker(model, p=numeric_norm, logger=self.logger, loss_f=None, track_acc=False)  # pytorch model

        # automatic batch size
        if batch_size is None:
            if device == "cpu":
                batch_size = 1
            else:
                batch_size = n_samples

        hyperparams = {**kwargs}
        hyperparams.pop("model")
        hyperparams.pop("dataset")
        hyperparams.pop("attack")
        hyperparams.pop("norm")
        hyperparams.pop("loss_f")


        data_loader = create_loaders(self.data_dir, task_config=kwargs["dataset"],
                                     batch_size=batch_size,
                                     transform=None,
                                     random_state=0,
                                     n_workers=n_workers,
                                    n_samples=n_samples)

        for b, (images, labels) in enumerate(tqdm(data_loader, total=n_samples // batch_size)):

            images, labels = images.to(device), labels.to(device)

            tracked_model.setup(images, labels)

            # run one clean run to get the clean accuracy
            tracked_model(images)
            
            # run attack
            if attack_f == aa:
                results = attack_f(model, images, labels, **norm_dict, **hyperparams)
                
                # track the results stats
                tracked_model(results)
            else:
                # run targeted attack 
                if "target" in hyperparams:
                    hyperparams["targeted"]=True
                    targets = torch.ones_like(labels)*hyperparams.pop("target")
                    
                    results = attack_f(tracked_model, images, targets, **norm_dict, **hyperparams)
                else:    
                    results = attack_f(tracked_model, images, labels, **norm_dict, **hyperparams)
                
            self.logger.concat_batch_log("results", [results])
            
            tracked_model.log()


    def save_df(self, path="logs/logmap.csv"):
        self.config_df.to_csv(path)


