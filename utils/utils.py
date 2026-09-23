import os
import random
import joblib
import numpy as np
import pandas as pd
import torch


class EarlyStopper:
    def __init__(self, patience=7, min_rel_delta=0.01):
        self.patience = patience
        self.min_rel_delta = min_rel_delta
        self.best = None
        self.bad_epochs = 0

    def step(self, val_metric):  # lower is better (e.g., MAE)
        if self.best is None or (self.best - val_metric) / max(self.best, 1e-8) > self.min_rel_delta:
            self.best = val_metric
            self.bad_epochs = 0
            return False  # don't stop
        else:
            self.bad_epochs += 1
            return self.bad_epochs >= self.patience


def _unwrap_model(model):
    # torch.compile wraps with _orig_mod; DataParallel/DDP wraps with .module
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model.module if hasattr(model, "module") else model

def save_training_state(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler = None,
    scaler: torch.cuda.amp.GradScaler = None,
    epoch: int = 0,
    global_step: int = 0,
    prev_losses: list[list[float]] = None,
    extra: dict = None,
):
    """
    Save everything needed to resume training exactly.
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    # RNG states for reproducibility on resume
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda_all"] = torch.cuda.get_rng_state_all()

    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "prev_losses": prev_losses,
        "model": _unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng_state": rng_state,
        "extra": extra or {},
        "pytorch_version": torch.__version__,
    }

    # Use a temporary file then atomic move to avoid partial writes
    tmp = filepath + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, filepath)

def load_training_state(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler = None,
    scaler: torch.cuda.amp.GradScaler = None,
    map_location = "cuda",
    strict: bool = True,
):
    """
    Load a checkpoint and restore model/optimizer/scheduler/scaler and RNG states.
    Returns a small dict with resume info: epoch, global_step, prev_losses, extra
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    # torch.load's weights_only=True exists in newer PyTorch; fall back if not
    try:
        ckpt = torch.load(filepath, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(filepath, map_location=map_location)

    # Model first
    _unwrap_model(model).load_state_dict(ckpt["model"], strict=strict)

    # Optimizer / scheduler / scaler if present
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except (ValueError, KeyError):
            print("Warning: scheduler state_dict incompatible (scheduler type changed?), starting scheduler fresh.")
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    # Restore RNG states (important for determinism when resuming mid-epoch)
    rng = ckpt.get("rng_state", {})
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("torch_cpu") is not None:
        torch.set_rng_state(rng["torch_cpu"].type(torch.ByteTensor))
    if torch.cuda.is_available() and rng.get("torch_cuda_all") is not None:
        torch.cuda.set_rng_state_all([x.type(torch.ByteTensor) for x in rng["torch_cuda_all"]])

    return {
        "epoch": ckpt.get("epoch", 0),
        "global_step": ckpt.get("global_step", 0),
        "prev_losses": ckpt.get("prev_losses", None),
        "extra": ckpt.get("extra", {}),
        "pytorch_version": ckpt.get("pytorch_version"),
    }

_FORECAST_CODE_MAP = {
    # code: (rain_intensity 0-1, cloud_cover 0-1)
    'FA': (0.0, 0.1),   # Fair (Day)
    'FW': (0.0, 0.1),   # Fair & Warm
    'PC': (0.0, 0.3),   # Partly Cloudy (Day)
    'LH': (0.0, 0.3),   # Slightly Hazy
    'WD': (0.0, 0.2),   # Windy
    'CL': (0.0, 0.7),   # Cloudy
    'HZ': (0.0, 0.5),   # Hazy
    'PS': (0.3, 0.5),   # Passing Showers
    'LS': (0.3, 0.6),   # Light Showers
    'LR': (0.3, 0.8),   # Light Rain
    'SH': (0.6, 0.8),   # Showers
    'RA': (0.6, 0.9),   # Moderate Rain
    'HS': (0.8, 0.9),   # Heavy Showers
    'HR': (0.8, 1.0),   # Heavy Rain
    'TL': (0.9, 1.0),   # Thundery Showers
    'HT': (0.95, 1.0),  # Heavy Thundery Showers
    'HG': (1.0, 1.0),   # Heavy Thundery Showers with Gusty Winds
}
SCALER_DIR = f"./scalers/"

def read_forecast_data(forecast_file_dir, device='cuda', use_scaler=False):
    _forecast_raw = pd.read_csv(forecast_file_dir)
    if use_scaler:
        temp_scaler = joblib.load(SCALER_DIR + 'temperature pt100.gz')
        rh_scaler = joblib.load(SCALER_DIR + 'relative humidity.gz')
        wspd_scaler = joblib.load(SCALER_DIR + 'wind speed.gz')
        for c in _forecast_raw.columns:
            if 'temperature' in c:
                _forecast_raw[c] = temp_scaler.transform(_forecast_raw[c].values.astype('float32').reshape(-1, 1)).reshape(-1)
            elif 'relative_humidity' in c:
                _forecast_raw[c] = rh_scaler.transform(_forecast_raw[c].values.astype('float32').reshape(-1, 1)).reshape(-1)
            elif 'wind_speed' in c:
                _forecast_raw[c] = wspd_scaler.transform(_forecast_raw[c].values.astype('float32').reshape(-1, 1)).reshape(-1)
    # numeric columns (temperature, humidity, wind, etc.)
    _num_cols = [c for c in _forecast_raw.columns if c != 'timestamp' and 'forecast_code' not in c]
    _forecast_vals = _forecast_raw[_num_cols].values.astype(np.float32)
    print(_forecast_vals[0])
    _forecast_vals = np.nan_to_num(_forecast_vals, nan=0.0)
    _fmin = _forecast_vals.min(axis=0, keepdims=True)
    _fmax = _forecast_vals.max(axis=0, keepdims=True)
    if not use_scaler:
        _forecast_vals = (_forecast_vals - _fmin) / (_fmax - _fmin + 1e-8)
    # forecast_code columns -> (rain_intensity, cloud_cover) per day
    _code_cols = sorted([c for c in _forecast_raw.columns if 'forecast_code' in c])
    _code_arrays = []
    for col in _code_cols:
        mapped = _forecast_raw[col].map(_FORECAST_CODE_MAP)
        rain = mapped.apply(lambda x: x[0] if isinstance(x, tuple) else 0.0).values.astype(np.float32)
        cloud = mapped.apply(lambda x: x[1] if isinstance(x, tuple) else 0.0).values.astype(np.float32)
        _code_arrays.extend([rain, cloud])
    _code_vals = np.stack(_code_arrays, axis=1)  # (T, n_days*2)
    # regroup into per-day blocks: (T, n_days, day_dim) where day_dim = numeric_per_day + 2 (rain, cloud)
    T = _forecast_raw.shape[0]
    n_days = len(_code_cols)                                   # e.g. 4
    num_per_day = _forecast_vals.shape[1] // n_days            # numeric features per day (e.g. 8)
    _forecast_vals = _forecast_vals.reshape(T, n_days, num_per_day)   # (T, n_days, num_per_day)
    _code_vals = _code_vals.reshape(T, n_days, 2)                     # (T, n_days, 2)
    _all_vals = np.concatenate([_forecast_vals, _code_vals], axis=2)  # (T, n_days, day_dim)
    return torch.tensor(_all_vals, device=device)
