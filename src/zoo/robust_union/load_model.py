from pathlib import Path
import os
import torch
from robustbench.utils import download_gdrive

from zoo.robust_union.preact_resnet import PreActResNet18
# from utils.config import DEVICE

MODEL_ID_MSD = '1CorITtTqTkB3N2D41NsyJB4W-Ql4HkGO'
MODEL_ID_AVG = '1dvlFR4iMDSIMLTRECc-G2-gRSe6Hcgd9'

def load_model_avg(dataset, norm, models_dir="models"):
    model = PreActResNet18()
    # base_path = Path(__file__).resolve()
    model_dir = f"{models_dir}/{dataset}/{norm}/AVG.pt"
    if not os.path.exists(model_dir):
        download_gdrive(MODEL_ID_AVG, model_dir)
    model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    model.eval()
    return model


def load_model_msd(dataset, norm, models_dir="models"):
    model = PreActResNet18()
    base_path = Path(__file__).resolve()
    # model_dir = base_path.parent / 'pretrained' / 'MSD.pt'
    model_dir = f"{models_dir}/{dataset}/{norm}/MSD.pt"
    if not os.path.exists(model_dir):
        download_gdrive(MODEL_ID_MSD, model_dir)
    model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    model.eval()
    return model
