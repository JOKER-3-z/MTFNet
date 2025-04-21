import torch
import numpy as np
import random

def set_seed(seed: int = 1) -> None:
    # 随机数种子设定
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def LoadParameter(_structure, _parameterDir,device='cpu'):
    checkpoint=torch.load(_parameterDir,map_location=device)
    pretrained_state_dict=checkpoint['state_dict']
    model_state_dict = _structure.state_dict()
    for key in pretrained_state_dict:
        if 'per_branch' in key:
            continue
        else:
            model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]
    _structure.load_state_dict(model_state_dict)
    return _structure