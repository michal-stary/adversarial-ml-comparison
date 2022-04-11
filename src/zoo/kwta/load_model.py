from pathlib import Path

import torch
from robustbench.utils import download_gdrive

from src.zoo.kwta import densenet
# from utils.config import DEVICE

MODEL_ID = '1tcepCx14tD5TaPs6KBjRT9n0Q8jjWDXj'

def load_model():
    sp = 0.1
    model = densenet.SparseDenseNet121(sparse_func='vol',
                                       sparsities=[sp, sp, sp, sp])
    base_path = Path(__file__).resolve()
    model_dir = base_path.parent / 'pretrained' / 'spDenseNet121_sp0.1_adv.pth'
    if not model_dir.exists():
        download_gdrive(MODEL_ID, model_dir)
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    return model
