# %%
import json
import math
import time
import numpy as np
import pandas as pd
import os
from random import randint

import torch
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt

import joblib

from utils.reproducibility import seed_everything, tf_func, clip_grad_func
from utils.loss_fn import masked_mae_multi as loss_fn
from utils.create_dataset_v1 import make_input_data, StationDataset
from utils.models import (AttrSeq2SeqLSTM as seq2seq_model, 
                          AttrSeq2SeqLSTM_v1 as seq2seq_model_v1,
                          AttrSeq2SeqCNNLSTM as seq2seq_cnn_model,
                          AttrLSTM as norm_model, 
                          AttrLSTMv1 as norm_model_v1, 
                          AttrSeq2SeqAttnLSTM as seq2seq_attn_model,
                          train_one_epoch, 
                          evaluate,
                          train_one_epoch_scaler,
                          evaluate_scaler,
                          model_run
                          )
from utils.utils import read_forecast_data

from sklearn.model_selection import train_test_split
from lstm1__const__ import DAYS, TRAIN_FILES, TEST_FILES


# %%
OUTPUT_FILE_NAME = f'_result_day_per_timept_07_27a'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

INP_ATTRS = None
OUT_ATTRS = None

INP_ATTR_SIZE = None
OUT_ATTR_SIZE = None

USE_SCALER = True

print(f'Using device: {DEVICE}')
ENCODER_IN_DIM = 0
DECODER_TF_DIM = 0

HIDDEN_DIM = 256

FORECAST_DATA = read_forecast_data('./data/forecast_hr.csv', device=DEVICE, use_scaler=USE_SCALER)
print('FORECAST_DATA.shape', FORECAST_DATA.shape)
# FORECAST_DATA: (T, n_days, day_dim) — forecast is grouped per day for per-step alignment
FORECAST_DAYS = FORECAST_DATA.shape[1]   # number of forecast days (e.g. 4)
FORECAST_DIM = FORECAST_DATA.shape[2]    # per-day feature dim (e.g. 10)
CNN_CONFIG = [[64, 128], 1, 1]

LOAD_EXISTING = True
EXISTING = f'./_result/_result_day_per_timept_07_27.json'

def _pick_best_epoch(val_losses, tol=0.0):
    v = np.asarray(val_losses, dtype=float)
    if v.size == 0:
        return None
    m = v.min()
    thresh = m * (1.0 + tol) if m > 0 else m + tol
    return int(np.argmax(v <= thresh))   # earliest epoch within tol (relative) of the min

def loadModel(DAY, loadbest=True):
    MODEL_LOADED = None
    epoch_files = {}
    for fname in os.listdir(f"models/{DAY}/"):
        if 'epoch_' not in fname:
            continue
        epoch_files[int(fname.split('epoch_')[1].split('.pth')[0])] = f"models/{DAY}/{fname}"
    filename = epoch_files[max(epoch_files)] if epoch_files else None
    def _load(path):
        try:
            return torch.load(path, map_location=DEVICE, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=DEVICE)
    if filename is not None:
        MODEL_LOADED = _load(filename)
        # Prefer the best-VALIDATION epoch (earliest at the min) over the last epoch, which for
        # these early-peaking models is the overfit tail. prev_losses is stored in the checkpoint.
        if loadbest:
            val = (MODEL_LOADED.get('prev_losses') or {}).get('test', {}).get('mean')
            if val:
                best = _pick_best_epoch(val)
                if best in epoch_files and epoch_files[best] != filename:
                    filename = epoch_files[best]
                    MODEL_LOADED = _load(filename)
                    print(f'  loadbest: epoch {best} (val {val[best]:.5f}) instead of last {max(epoch_files)}')
        extra = MODEL_LOADED['extra']
        INP_ATTRS = extra['inputs']
        OUT_ATTRS = extra['outputs']
        ENCODER_IN_DIM = extra['encoder_in_dim']
        DECODER_TF_DIM = extra['decoder_tf_dim']
        USE_SCALER = extra['use_scaler']
        LOOKBACK = extra['lookback']
        LOOKFORWARD = extra['lookforward']
        DROPOUT = extra['dropout']
        USE_WAVELET = extra['use_wavelet'] if 'use_wavelet' in extra else USE_WAVELET
        WAVELET_LVL = extra['wavelet_lvl'] if 'wavelet_lvl' in extra else WAVELET_LVL
        STN_MATCH_ATTRIBUTE = extra['stn_match_attr'] if 'stn_match_attr' in extra else STN_MATCH_ATTRIBUTE
        cnn_config = extra['cnn_config'] if 'cnn_config' in extra else CNN_CONFIG
        forecast_dim = extra['forecast_dim'] if 'forecast_dim' in extra else None
        HIDDEN_DIM = extra['hidden_dim'] if 'hidden_dim' in extra else HIDDEN_DIM
        OUT_ATTR_SIZE = len(OUT_ATTRS)
        model = None
        print('model type:', extra['model_type'], forecast_dim)
        if 'model_type' in extra and extra['model_type'] == 'seq2seq':
            model = seq2seq_model(ENCODER_IN_DIM, DECODER_TF_DIM, OUT_ATTR_SIZE, hidden_dim=HIDDEN_DIM)
        elif 'model_type' in extra and extra['model_type'] == 'seq2seq_v1':
            model = seq2seq_model_v1(ENCODER_IN_DIM, DECODER_TF_DIM, OUT_ATTR_SIZE, hidden_dim=HIDDEN_DIM, forecast_dim=forecast_dim, forecast_days=FORECAST_DAYS)
        elif 'model_type' in extra and extra['model_type'] == 'seq2seq_attn':
            model = seq2seq_attn_model(ENCODER_IN_DIM, DECODER_TF_DIM, OUT_ATTR_SIZE, hidden_dim=HIDDEN_DIM, forecast_dim=forecast_dim, forecast_days=FORECAST_DAYS)
        elif 'model_type' in extra and extra['model_type'] == 'seq2seq_cnn':
            model = seq2seq_cnn_model(
                ENCODER_IN_DIM, DECODER_TF_DIM, OUT_ATTR_SIZE,
                hidden_dim=HIDDEN_DIM,
                forecast_dim=forecast_dim, forecast_days=FORECAST_DAYS,
                # CNN experiments (toggle here):
                #   current 3-layer pooling : cnn_channels=(64,128,256), pool_size=2, dilation=1
                #   2-layer pooling         : cnn_channels=(64,128),     pool_size=2, dilation=1
                #   dilated, no pooling     : cnn_channels=(64,128),     pool_size=1, dilation=2
                cnn_channels=cnn_config[0], pool_size=cnn_config[1], dilation=cnn_config[2],
            )
        elif 'model_type' in extra and extra['model_type'] == 'plain_lstm':
            model = norm_model(ENCODER_IN_DIM, DECODER_TF_DIM, OUT_ATTR_SIZE, hidden_dim=HIDDEN_DIM, forecast_dim=forecast_dim, forecast_days=FORECAST_DAYS)
        elif 'model_type' in extra and extra['model_type'] == 'v1':
            model = norm_model_v1(ENCODER_IN_DIM, DECODER_TF_DIM, OUT_ATTR_SIZE, hidden_dim=HIDDEN_DIM)
        else:
            model = norm_model(ENCODER_IN_DIM, DECODER_TF_DIM, OUT_ATTR_SIZE, hidden_dim=HIDDEN_DIM)

        # model.load_state_dict(MODEL_LOADED, strict=True)
        state_dict = MODEL_LOADED["model"]
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k[len("_orig_mod."):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        # _unwrap_model(model).load_state_dict(MODEL_LOADED["model"], strict=True)
        prev_losses = MODEL_LOADED["prev_losses"]

        print('Model loaded:', filename)
    model.to(DEVICE) 
    return {'model': model, 'model_info': MODEL_LOADED}
LOADED = 1
del LOADED


# ---- non-learned baselines, added as extra "models" so they land in the same results table ----
ADD_BASELINES = True
BASELINES = [('Persistence', 'persistence'),
             ('Persistence24', 'persistence24'),
             ('Climatology', 'climatology')]

def _hour_bin(sin, cos):
    # recover hour-of-day (0-23) from [minute_sin, minute_cos]
    frac = torch.remainder(torch.atan2(sin, cos) / (2 * math.pi), 1.0)
    return torch.remainder(torch.round(frac * 24).long(), 24)

class Baseline:
    """Non-learned baseline that mimics a model for the eval loop.
       persistence   : repeat the last observed value across the whole horizon. Flat-hold;
                       only a meaningful reference for the first few lead hours.
       persistence24 : seasonal-naive with period 24 — replay the last observed 24h cycle,
                       matched by hour-of-day, so it tiles across all 7 forecast days. This
                       is the baseline that actually competes on a diurnal signal.
       climatology   : per-sample masked mean of the ENCODER history at the same
                       hour-of-day as each forecast step (recent diurnal climatology,
                       computed from the station's own recent window -> works for
                       held-out test stations too, no cross-station leakage).
       persistence24 and climatology share one code path: both are an hour-of-day mean of the
       encoder history, differing only in how far back the window reaches (24h vs the full
       lookback). With a 24h window each hour bin holds at most one observation, so the mean
       IS that hour's last observed value.
       Values live in the same scaled space as the model outputs, so downstream
       inverse-scaling and metrics are identical."""
    is_baseline = True
    def __init__(self, kind, extra):
        self.kind = kind
        self.model_type = f'baseline_{kind}'
        inputs = [a.lower() for a in extra['inputs']]
        outs   = [a.lower() for a in extra['outputs']]
        K = len(inputs)
        self.val_ch = [5 + K + inputs.index(a) for a in outs]   # target value channels in series
        self.msk_ch = [5 + inputs.index(a) for a in outs]        # target presence-mask channels
        self.out_len = extra['lookforward']
        self.window  = 24 if kind == 'persistence24' else None   # None = whole lookback
    def eval(self):            return self
    def to(self, *a, **k):     return self
    def predict(self, enc_seq, dec_tf, last_val):
        B = enc_seq.shape[0]
        C = len(self.val_ch)
        if self.kind == 'persistence':
            return last_val[:, :1, :].expand(B, self.out_len, C).contiguous()
        vals   = enc_seq[:, :, self.val_ch]                       # (B, L, C)
        msk    = enc_seq[:, :, self.msk_ch]                       # (B, L, C) 1=present
        enc_hr = _hour_bin(enc_seq[:, :, 0], enc_seq[:, :, 1])    # (B, L)
        dec_hr = _hour_bin(dec_tf[:, :, 0],  dec_tf[:, :, 1])     # (B, T)
        if self.window is not None:
            vals, msk, enc_hr = vals[:, -self.window:], msk[:, -self.window:], enc_hr[:, -self.window:]
        L = vals.shape[1]
        idx  = enc_hr.unsqueeze(-1).expand(B, L, C)
        csum = torch.zeros(B, 24, C, device=vals.device, dtype=vals.dtype).scatter_add_(1, idx, vals * msk)
        ccnt = torch.zeros(B, 24, C, device=vals.device, dtype=vals.dtype).scatter_add_(1, idx, msk)
        overall = (vals * msk).sum(1, keepdim=True) / msk.sum(1, keepdim=True).clamp_min(1e-6)   # (B,1,C)
        clim = torch.where(ccnt < 0.5, overall.expand(B, 24, C), csum / ccnt.clamp_min(1e-6))    # (B,24,C)
        gidx = dec_hr.unsqueeze(-1).expand(B, self.out_len, C)
        return torch.gather(clim, 1, gidx)                        # (B, T, C)


# %%
print('......', DAYS.keys().__iter__().__next__(), DAYS[DAYS.keys().__iter__().__next__()])

try:
    print(LOADED)
except Exception:
    MODELS  = {}
    RESULTS = {}
    for i0 in DAYS:
        if i0 not in MODELS:
            MODELS[i0] = {}
            RESULTS[i0] = {}
        for i1 in DAYS[i0]:
            print(i0, i1, DAYS[i0][i1])
            MODELS[i0][i1] = loadModel(DAYS[i0][i1])
            RESULTS[i0][i1] = {}

    # Add non-learned baselines as extra "models", one per attribute, reusing a loaded
    # model's data config (extra) so they share the same datasets/scaling/splits.
    if ADD_BASELINES:
        _attrs_ref = {}
        for _i0 in list(MODELS):
            for _a in MODELS[_i0]:
                _attrs_ref.setdefault(_a, MODELS[_i0][_a]['model_info'])
        for _name, _kind in BASELINES:
            MODELS.setdefault(_name, {}); RESULTS.setdefault(_name, {}); DAYS.setdefault(_name, {})
            for _a, _ref in _attrs_ref.items():
                _extra = dict(_ref['extra'])
                _extra['model_type'] = f'baseline_{_kind}'
                _extra['use_wavelet'] = False
                MODELS[_name][_a] = {'model': Baseline(_kind, _extra), 'model_info': {'extra': _extra}}
                DAYS[_name][_a] = _name
                RESULTS[_name][_a] = {}

    if LOAD_EXISTING:
        with open(EXISTING) as f0: _prev = json.loads(f0.read())
        # Merge (don't replace): keep the freshly-built RESULTS keys, including the baselines,
        # and copy ONLY train/eval from the existing file where present. 'test' is left absent
        # so evaluate_models recomputes it; models missing from the old file (baselines/new)
        # simply have no prior train/eval and get all three computed fresh.
        for i0 in RESULTS:
            for i1 in RESULTS[i0]:
                _p = _prev.get(i0, {}).get(i1, {})
                for _split in ('train', 'eval', 'test'):
                    if _split in _p:
                        RESULTS[i0][i1][_split] = _p[_split]
                        print('......', i0, i1, _split)

    LOADED = 'MODEL AND DATA LOADED'


# %%
DATA_DIR = 'model_data_1/'
FILE_SUFFIX = '_hr_avg.csv'
# TRAIN_FILES = [
#     ############ train stations ############
#     '21004880', '21024003', '21026652', 
#     '21027656', '21027657', '21027658', 
#     '21027660', '21027661', 
#     '21027662', '21027663', 
#     '21027665', ## Singapore General Hospital #4
#     '21040276', '21040277', 
#     '21040278', '21040279', '21040280', '23001059', 
#     '23003762', '23003763', '23003764', '23003765', '23003766', 

#     'S100', 'S102', 'S104', 'S106', 
#     'S108', 'S109', 
#     'S111', 'S115', 'S116', 'S117', 
#     'S24', 
#     'S43', 'S44', 'S60', 
# ]
# TEST_FILES = [
#     ############ test stations #############
#     '21027659', # Pinnacle@ Duxton_Everton Lamppost #8
#     '23001060', # Fuhua School mid-lvl big podium #4
#     '21026653', # Pinnacle@ Duxton_Ground Along Road #5
#     # '21027665', ## Singapore General Hospital #4
#     'S06', 
#     'S50', # Clementi Road
#     'S107', # East Coast Parkway
#     'S121', # Old Choa Chu Kang Road

#     # # obsoletes: 'S24B', 'S96', 'S97', 'S122'
# ] 


# # LOOKBACK_DAYS = 60
# # INTERVAL = 60 * 60
# # LOOKBACK_TIME = 60 * 60 * 24 * LOOKBACK_DAYS
# # LOOKBACK = round(LOOKBACK_TIME / INTERVAL)
# LOOKBACK = None

# # LOOKFORWARD_TIME = 60 * 60 * 24 * 7
# # LOOKFORWARD = round(LOOKFORWARD_TIME / INTERVAL)
# LOOKFORWARD = None

BATCH_SIZE = 256
BATCH_SIZE_EFF = 256
BATCH_ACCU = int(BATCH_SIZE_EFF / BATCH_SIZE)
BATCH_COUNT = int(BATCH_SIZE_EFF * math.floor(5000 / BATCH_SIZE_EFF))


# INP_ATTRS = None
# OUT_ATTRS = None

# INP_ATTR_SIZE = None
# OUT_ATTR_SIZE = None

# USE_SCALER = False

# print(f'Using device: {DEVICE}')
# ENCODER_IN_DIM = 0
# DECODER_TF_DIM = 0

# HIDDEN_DIM = 256


SEED = 42
STN_MATCH_ATTRIBUTE = 'lat_long_h'

ATTR_MATCH = {
    'temperature pt100': 'Temp',
    'relative humidity': 'RH',
    'wind speed': 'WSpd',
    'sin_wind': 'Wdir',
    'cos_wind': 'Wdir',
    'sin_dir': 'Wdir',
    'sin_dir': 'Wdir',
    'solar radiation': 'Sol',
    'pm1.0': 'pm1',
    'pm2.5': 'pm2p5',
    'pm10': 'pm10'
}

SCALER = {}
INP_ATTRS = ['Temperature PT100', 'Relative Humidity', 'Wind speed', 'sin_dir', 'cos_dir', 'Solar radiation']

for attr in INP_ATTRS:
    p = f"./scalers/{attr}.gz"
    if os.path.exists(p):
        SCALER[attr.lower()] = joblib.load(p)
        print(f'scaler file - OK: ./scalers/{attr}.gz')
    else:
        SCALER[attr.lower()] = None
        print(f'scaler file - not exist: ./scalers/{attr}.gz')

stn_match_file = open(DATA_DIR + 'station_match_1.json')
station_match = json.loads(stn_match_file.read())
stn_match_file.close()


# %%
def _build_station_datasets(MODEL_DATA):
    """Build per-station StationDataset objects for the train/eval/test splits.

    Returns three dicts keyed by station file id:
      - train_ds / eval_ds : each TRAIN_FILES station split 70/30 (train/eval)
      - test_ds            : each TEST_FILES station (held out entirely)
    """
    extra = MODEL_DATA['extra']
    OUT_ATTRS = extra['outputs']
    LOOKBACK = extra['lookback']
    LOOKFORWARD = extra['lookforward']
    USE_WAVELET = extra['use_wavelet'] if 'use_wavelet' in extra else USE_WAVELET
    WAVELET_LVL = extra['wavelet_lvl'] if 'wavelet_lvl' in extra else WAVELET_LVL
    BATCHNORM = extra['batchnorm'] if 'batchnorm' in extra else BATCHNORM
    STN_MATCH_ATTRIBUTE = extra['stn_match_attr'] if 'stn_match_attr' in extra else STN_MATCH_ATTRIBUTE
    # Must match how the model was trained: if it was trained without the absolute-year feature,
    # zero year_norm here too. Default True for older checkpoints (trained with year_norm included).
    INCLUDE_YEAR = extra.get('include_year', True)

    DATA_DIR = 'model_data_preprocessed_nosol/' + ATTR_MATCH[OUT_ATTRS[0]] + '_60/'

    train_ds = {}
    eval_ds = {}
    test_ds = {}

    for file in TRAIN_FILES:
        stn_data = np.load(f'{DATA_DIR}/{file}.npz')
        stn_info = np.array(station_match[file][STN_MATCH_ATTRIBUTE], dtype=np.float32)
        time_feats, target, target_mask, valid_idx = stn_data['time_feats'], stn_data['target'], stn_data['target_mask'], stn_data['valid_idx']
        series = stn_data['series']
        inp = stn_data['input']
        inp_mask = stn_data['input_mask']
        forecast_idx = stn_data['forecast_idx']
    
        # Date-based split on the forecast target period: TEST = windows whose 168h target
        # starts on/after SPLIT_DATE; TRAIN = windows whose target ends before SPLIT_DATE.
        # Windows straddling the boundary are dropped so no training target overlaps the eval
        # period. Exact for a Jan-1 cutoff: time_feats[:, 4] is the absolute year (years since
        # 1970, base_year=0), so the year comparison IS the date comparison.
        SPLIT_DATE = np.datetime64('2025-01-01')
        SPLIT_YEAR = int(SPLIT_DATE.astype('datetime64[Y]').astype(int))  # years since 1970; exact for a Jan-1 cutoff
        year_of = time_feats[:, 4]
        tgt_start_year = year_of[valid_idx + LOOKBACK]
        tgt_end_year   = year_of[valid_idx + LOOKBACK + LOOKFORWARD - 1]
        is_eval  = tgt_start_year >= SPLIT_YEAR
        is_train = tgt_end_year   <  SPLIT_YEAR
        train_idx, train_forecast = valid_idx[is_train], forecast_idx[is_train]
        eval_idx,  eval_forecast  = valid_idx[is_eval],  forecast_idx[is_eval]
        if not INCLUDE_YEAR:
            time_feats[:, 4] = 0.0   # series[:, :5] IS time_feats, so zero both
            series[:, 4] = 0.0

        train_ds[file] = StationDataset(
            series, time_feats, target, target_mask, train_idx, stn_info,
            LOOKBACK, LOOKFORWARD, forecast=FORECAST_DATA, forecast_idx=train_forecast,
            use_wavelet=USE_WAVELET, wavelet_level=WAVELET_LVL, inp=inp, inp_mask=inp_mask)
        eval_ds[file] = StationDataset(
            series, time_feats, target, target_mask, eval_idx, stn_info,
            LOOKBACK, LOOKFORWARD, forecast=FORECAST_DATA, forecast_idx=eval_forecast,
            use_wavelet=USE_WAVELET, wavelet_level=WAVELET_LVL, inp=inp, inp_mask=inp_mask)

    for file in TEST_FILES:
        stn_data = np.load(f'{DATA_DIR}/{file}.npz')
        stn_info = np.array(station_match[file][STN_MATCH_ATTRIBUTE], dtype=np.float32)
        time_feats, target, target_mask, valid_idx = stn_data['time_feats'], stn_data['target'], stn_data['target_mask'], stn_data['valid_idx']
        series = stn_data['series']
        inp = stn_data['input']
        inp_mask = stn_data['input_mask']
        forecast_idx = stn_data['forecast_idx']

        # Date-based split on the forecast target period: TEST = windows whose 168h target
        # starts on/after SPLIT_DATE; TRAIN = windows whose target ends before SPLIT_DATE.
        # Windows straddling the boundary are dropped so no training target overlaps the eval
        # period. Exact for a Jan-1 cutoff: time_feats[:, 4] is the absolute year (years since
        # 1970, base_year=0), so the year comparison IS the date comparison.
        SPLIT_DATE = np.datetime64('2025-01-01')
        SPLIT_YEAR = int(SPLIT_DATE.astype('datetime64[Y]').astype(int))  # years since 1970; exact for a Jan-1 cutoff
        year_of = time_feats[:, 4]
        tgt_start_year = year_of[valid_idx + LOOKBACK]
        tgt_end_year   = year_of[valid_idx + LOOKBACK + LOOKFORWARD - 1]
        is_eval  = tgt_start_year >= SPLIT_YEAR
        eval_idx,  eval_forecast  = valid_idx[is_eval],  forecast_idx[is_eval]

        if not INCLUDE_YEAR:
            time_feats[:, 4] = 0.0   # series[:, :5] IS time_feats, so zero both
            series[:, 4] = 0.0
        test_ds[file] = StationDataset(
            series, time_feats, target, target_mask, eval_idx, stn_info,
            LOOKBACK, LOOKFORWARD, forecast=FORECAST_DATA, forecast_idx=eval_forecast,
            use_wavelet=USE_WAVELET, wavelet_level=WAVELET_LVL, inp=inp, inp_mask=inp_mask)

    return train_ds, eval_ds, test_ds


def createDataloader(MODEL_DATA):
    train_ds, eval_ds, test_ds = _build_station_datasets(MODEL_DATA)

    # Per-station dataloaders (keyed by station file id)
    train_station_loaders = {f: data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False) for f, ds in train_ds.items()}
    eval_station_loaders = {f: data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False) for f, ds in eval_ds.items()}
    test_station_loaders = {f: data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False) for f, ds in test_ds.items()}

    # Full (concatenated) datasets/dataloaders across all stations of the split
    full_train_dataset = data.ConcatDataset(list(train_ds.values()))
    full_eval_dataset = data.ConcatDataset(list(eval_ds.values()))
    full_test_dataset = data.ConcatDataset(list(test_ds.values()))
    train_dataloader = data.DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    eval_dataloader = data.DataLoader(full_eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = data.DataLoader(full_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return {
        'train': train_dataloader,
        'eval': eval_dataloader,
        'test': test_dataloader,
        'train_ds': full_train_dataset,
        'eval_ds': full_eval_dataset,
        'test_ds': full_test_dataset,
        # per-station dataloaders
        'train_stations': train_station_loaders,
        'eval_stations': eval_station_loaders,
        'test_stations': test_station_loaders,
    }


# ----- streaming metric accumulators (MAE / RMSE / R2) -----
# Metrics are computed PER TIMESTEP: every valid (sample, timestep) value is an
# observation, so count = number of valid timesteps. R2 uses a Welford-style
# running total so per-station accumulators can be merged into the full-dataset one.

def _new_acc():
    return {'n': 0, 'sum_ae': 0.0, 'sum_se': 0.0, 'mean_a': 0.0, 'ss_tot': 0.0}


def _update_acc(acc, a, r):
    """Fold a chunk of per-sample actual (a) and predicted (r) values into acc."""
    m = len(a)
    if m == 0:
        return
    errors = r - a
    acc['sum_ae'] += float(np.sum(np.abs(errors)))
    acc['sum_se'] += float(np.sum(errors ** 2))
    n = acc['n']
    chunk_mean = float(a.mean())
    chunk_ss = float(np.sum((a - chunk_mean) ** 2))
    delta = chunk_mean - acc['mean_a']
    n_new = n + m
    acc['ss_tot'] += chunk_ss + delta ** 2 * n * m / n_new
    acc['mean_a'] = (n * acc['mean_a'] + m * chunk_mean) / n_new
    acc['n'] = n_new


def _merge_acc(dst, src):
    """Merge accumulator src into dst (combining two Welford totals)."""
    m = src['n']
    if m == 0:
        return
    n = dst['n']
    if n == 0:
        dst.update(src)
        return
    n_new = n + m
    delta = src['mean_a'] - dst['mean_a']
    dst['ss_tot'] += src['ss_tot'] + delta ** 2 * n * m / n_new
    dst['mean_a'] = (n * dst['mean_a'] + m * src['mean_a']) / n_new
    dst['sum_ae'] += src['sum_ae']
    dst['sum_se'] += src['sum_se']
    dst['n'] = n_new


def _finalize_acc(acc):
    n = acc['n']
    if n <= 0:
        return {}
    return {
        'mae': float(acc['sum_ae'] / n),
        'rmse': float(np.sqrt(acc['sum_se'] / n)),
        'r2': float(1 - acc['sum_se'] / acc['ss_tot']) if acc['ss_tot'] > 0 else float('nan'),
        'count': int(n),
    }


def _finalize_days(day_accs):
    """Finalize a {day_idx: acc} dict into {day_idx: {mae, rmse, r2, count}}."""
    return {d: _finalize_acc(a) for d, a in sorted(day_accs.items()) if a['n'] > 0}


def _merge_day_accs(dst, src):
    """Merge a {day_idx: acc} dict src into dst."""
    for d, a in src.items():
        if d not in dst:
            dst[d] = _new_acc()
        _merge_acc(dst[d], a)


def _forecast_day_index(dec_tf, B, T):
    """Per-timestep forecast-day index (B, T) derived from decoder time features.

    A day boundary (local midnight) is where sin(minute-of-day)==0 and
    cos(minute-of-day)==1, i.e. dec_tf[..., 0]==0 and dec_tf[..., 1]==1
    (noon also has sin==0 but cos==-1, so requiring cos==1 disambiguates).
    Day 0 covers the steps from the forecast start up to the first midnight;
    each midnight starts the next day (day 1 = next calendar day, ... day 7).
    """
    dt = dec_tf.detach().cpu().float().numpy().reshape(B, dec_tf.shape[1], -1)[:, :T, :]
    sin_m = dt[:, :, 0]
    cos_m = dt[:, :, 1]
    is_midnight = np.isclose(sin_m, 0.0, atol=1e-4) & np.isclose(cos_m, 1.0, atol=1e-4)
    return np.cumsum(is_midnight, axis=1).astype(np.int64)   # (B, T)


def _masked_values(actual, result, sample_mask):
    """Per-timestep valid values: every (sample, timestep) where sample_mask is True
    becomes one observation. Returns flat (a, r) arrays."""
    return actual[sample_mask], result[sample_mask]


def _circular_values(sinA, cosA, sinR, cosR, sample_mask):
    """Per-timestep direction (degrees) for wind direction.

    Converts each valid (sample, timestep) sin/cos pair to an angle and returns
    (a, r) where a = actual angle and r = a + wrapped(pred - actual). Feeding
    (a, r) to _update_acc makes (r - a) the wrap-around error in [-180, 180),
    so MAE/RMSE are in degrees and R2 is measured against actual-direction
    variance.
    """
    a_ang = np.degrees(np.arctan2(sinA, cosA)) % 360.0
    r_ang = np.degrees(np.arctan2(sinR, cosR)) % 360.0
    err = (r_ang - a_ang + 180.0) % 360.0 - 180.0
    return a_ang[sample_mask], (a_ang + err)[sample_mask]


def _accumulate_loader(model_type, model, dataloader, acc, day_accs, scaler, apply_scaler):
    """Run inference over one dataloader, folding results into:
      - acc      : all-steps per-timestep metric accumulator
      - day_accs : {forecast_day_index: acc} per-day, per-timestep metric accumulators

    Wind direction (model_type == 'WDir') is handled as circular angles: its two
    output channels are [sin(dir), cos(dir)] and metrics use wrap-around error.
    """
    model.eval()
    is_wdir = str(model_type).lower() == 'wdir'
    is_baseline = getattr(model, 'is_baseline', False)
    for enc_seq, dec_tf, mask, y, stn, last_val, forecast in dataloader:
        if is_baseline:
            with torch.no_grad():
                pred = model.predict(enc_seq, dec_tf, last_val)
        else:
            with torch.no_grad(), torch.autocast('cuda'):
                pred = model_run(model, enc_seq, dec_tf, 0.0, last_val=last_val, station_feats=stn, forecast=forecast)
        B = y.shape[0]

        if is_wdir:
            sinA = y[:, :, 0].cpu().float().numpy().astype(np.float64)               # (B, T)
            cosA = y[:, :, 1].cpu().float().numpy().astype(np.float64)
            sinR = pred[:, :, 0].detach().cpu().float().numpy().astype(np.float64)
            cosR = pred[:, :, 1].detach().cpu().float().numpy().astype(np.float64)
            T = sinA.shape[1]
            valid = mask.cpu().numpy().reshape(B, T, -1)[..., 0].astype(bool)        # (B, T)

            # all-steps circular metrics (per timestep)
            a, r = _circular_values(sinA, cosA, sinR, cosR, valid)
            _update_acc(acc, a, r)

            # per forecast-day circular metrics (per timestep)
            day_idx = _forecast_day_index(dec_tf, B, T)
            for d in np.unique(day_idx):
                d = int(d)
                day_valid = valid & (day_idx == d)
                if not day_valid.any():
                    continue
                ad, rd = _circular_values(sinA, cosA, sinR, cosR, day_valid)
                if len(ad) == 0:
                    continue
                if d not in day_accs:
                    day_accs[d] = _new_acc()
                _update_acc(day_accs[d], ad, rd)
            continue

        # (B, T) — collapse any trailing attr dim into T
        actual = y.cpu().float().numpy().reshape(B, -1).astype(np.float64)
        result = pred.detach().cpu().float().numpy().reshape(B, -1).astype(np.float64)
        valid = mask.cpu().numpy().reshape(B, -1).astype(bool)   # (B, T)
        T = actual.shape[1]
        if apply_scaler:
            actual = scaler.inverse_transform(actual.reshape(-1, 1)).reshape(B, -1)
            result = scaler.inverse_transform(result.reshape(-1, 1)).reshape(B, -1)

        # ----- all-steps per-timestep masked values -----
        a, r = _masked_values(actual, result, valid)
        _update_acc(acc, a, r)

        # ----- per forecast-day metrics (per timestep) -----
        day_idx = _forecast_day_index(dec_tf, B, T)              # (B, T)
        for d in np.unique(day_idx):
            d = int(d)
            day_valid = valid & (day_idx == d)
            if not day_valid.any():
                continue
            ad, rd = _masked_values(actual, result, day_valid)
            if len(ad) == 0:
                continue
            if d not in day_accs:
                day_accs[d] = _new_acc()
            _update_acc(day_accs[d], ad, rd)


def evaluate_model_dataset(model_type, model, station_loaders, out_attrs, use_scaler=True):
    """Evaluate a model over per-station dataloaders.

    `station_loaders` is a dict {station_id: DataLoader}. Returns:
        {'full':     {mae, rmse, r2, count, 'days': {d: {mae, rmse, r2, count}}},
         'stations': {station_id: {mae, rmse, r2, count, 'days': {...}}, ...}}
    where 'full' aggregates over every station, and 'days' breaks the metrics
    down by forecast-day index (0 = same day until midnight, 1 = next day, ...).
    """
    scaler = SCALER[out_attrs[0]] if out_attrs[0] in SCALER else None
    apply_scaler = use_scaler and scaler is not None and len(out_attrs) == 1

    full_acc = _new_acc()
    full_day_accs = {}
    per_station = {}
    start0 = time.perf_counter()
    for station, dataloader in station_loaders.items():
        start = time.perf_counter()
        acc = _new_acc()
        day_accs = {}
        _accumulate_loader(model_type, model, dataloader, acc, day_accs, scaler, apply_scaler)
        st = _finalize_acc(acc)
        if st:
            st['days'] = _finalize_days(day_accs)
        per_station[station] = st
        _merge_acc(full_acc, acc)
        _merge_day_accs(full_day_accs, day_accs)
        if st:
            print(f'  [{model_type}] {station}: MAE={st["mae"]:#.6f}, RMSE={st["rmse"]:#.6f}, '
                  f'R2={st["r2"]:#.4f}, n={st["count"]:,} ({time.perf_counter()-start:.2f}s)')

    full = _finalize_acc(full_acc)
    if full:
        full['days'] = _finalize_days(full_day_accs)
        print(f'  [{model_type}] FULL: MAE={full["mae"]:#.6f}, RMSE={full["rmse"]:#.6f}, '
              f'R2={full["r2"]:#.4f}, n={full["count"]:,} ({time.perf_counter()-start0:.2f}s)')
        for d in sorted(full['days']):
            dm = full['days'][d]
            print(f'      day {d}: MAE={dm["mae"]:#.6f}, RMSE={dm["rmse"]:#.6f}, '
                  f'R2={dm["r2"]:#.4f}, n={dm["count"]:,}')
    return {'full': full, 'stations': per_station}


# %%
def evaluate_models():
    for i0 in DAYS:
        for i1 in DAYS[i0]:
            model = MODELS[i0][i1]['model']
            model_data = MODELS[i0][i1]['model_info']
            out_attrs = model_data['extra']['outputs']
            use_scaler = model_data['extra']['use_scaler']
            result = RESULTS[i0][i1]

            model.eval()
            dataloaders = createDataloader(model_data)

            # Each result is {'full': {mae, rmse, r2, count}, 'stations': {stn: {...}}}.
            # TRAIN_FILES stations appear in both 'train' and 'eval' (their 70/30 split);
            # TEST_FILES stations appear only in 'test'.

            # Reuse train/eval from the loaded results if present; only compute them when
            # missing (e.g. baselines or models not in the existing file). Always recompute test.
            if 'train' not in result:
                print(f'=== {i0} / {i1} : TRAIN ===')
                result['train'] = evaluate_model_dataset(i1, model, dataloaders['train_stations'], out_attrs, use_scaler=use_scaler)
            if 'eval' not in result:
                print(f'=== {i0} / {i1} : EVAL ===')
                result['eval']  = evaluate_model_dataset(i1, model, dataloaders['eval_stations'], out_attrs, use_scaler=use_scaler)
            if 'test' not in result:
                print(f'=== {i0} / {i1} : TEST ===')
                result['test']  = evaluate_model_dataset(i1, model, dataloaders['test_stations'], out_attrs, use_scaler=use_scaler)
            print('train full:', result['train']['full'])
            print('eval  full:', result['eval']['full'])
            print('test  full:', result['test']['full'])
            # return
        
evaluate_models()

# %%
with open(f'./_result/{OUTPUT_FILE_NAME}.json','w') as f: f.write(json.dumps(RESULTS))
