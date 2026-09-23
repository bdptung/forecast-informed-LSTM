# reproducibility.py
import os, random, numpy as np, torch

def seed_everything(seed: int = 42, deterministic: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuDNN / kernels
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)  # error if an op is nondeterministic
        # Make cuBLAS deterministic (needed for some GEMMs)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"
        # For bitwise-stable math (optional—slower but safer)
        # torch.backends.cuda.matmul.allow_tf32 = False
        # torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.conv.fp32_precision = False

def tf_func(epoch):
    return 0
    return max(0.1, 0.9 - epoch*0.08)

def clip_grad_func(epoch):
    return 1.0
    return max(1.0, 5.0 - epoch*2)

