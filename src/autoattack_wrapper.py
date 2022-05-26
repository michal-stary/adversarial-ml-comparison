from autoattack import AutoAttack
from torch import nn, Tensor
import torch
from tracking import PyTorchModelTracker

def aa(model: nn.Module,
       inputs: Tensor,
       labels: Tensor,
       norm,
       eps
      ):
    
    adversary = AutoAttack(model, norm=norm, eps=eps, log_path='./log_file.txt',
                            version="standard")
    
    adv_complete = adversary.run_standard_evaluation(inputs, labels,
                bs=len(inputs))
    
    return adv_complete


