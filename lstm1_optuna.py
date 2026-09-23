import json
import numpy as np
import os

import torch
import torch.optim as optim
import torch.utils.data as data

import joblib
import optuna
from sklearn.model_selection import train_test_split

from utils.reproducibility import seed_everything
from utils.loss_fn import LossFN
from utils.create_dataset_v1 import StationDataset
from utils.models import (AttrSeq2SeqCNNLSTM as seq2seq_cnn_model,
                          train_one_epoch_scaler_v1,
                          evaluate_v1)
from utils.utils import read_forecast_data, save_training_state

SEED = 42
seed_everything(SEED, False)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#####################################################################################
DATA_DIR_GEN = 'model_data_preprocessed_0/_general/'
DATA_DIR = 'model_data_preprocessed_0/WSpd_60'
DATA_DIR_GEN = DATA_DIR
if not DATA_DIR.endswith('/'): DATA_DIR += '/'

BATCH_SIZE      = 256
N_OPTUNA_EPOCHS = 15      # full-length trials — each trial is a complete training run
N_TRIALS        = 6      # fewer trials since each is now expensive
MAX_TRAIN       = 5000    # per station — bump toward 21000 if the saved model must be directly deployable
MAX_TEST        = 3000
# MAX_TRAIN = 21000
# MAX_TEST  = 9000

LOSS_TYPE           = 'mae'
STN_MATCH_ATTRIBUTE = 'lat_long_h_scaled'
LOSS_GRAD_WEIGHT    = 0
USE_FORECAST        = True
USE_WAVELET         = True
WAVELET_LVL         = 4
MODEL_TYPE          = 'seq2seq_cnn'

FILE_SUFFIX = '_hr_avg.csv'
FILES = [
    '21004880', '21024003', '21026652',
    '21027656', '21027657', '21027658',
    '21027660', '21027661',
    '21027662', '21027663', '21027665', '21040276', '21040277',
    '21040278', '21040279', '21040280', '23001059',
    '23003762', '23003763', '23003764', '23003765', '23003766',
    'S100', 'S102', 'S104', 'S106',
    'S108', 'S109',
    'S111', 'S115', 'S116', 'S117',
    'S122', 'S24',
    'S43', 'S44', 'S60',
]

with open(DATA_DIR + '_info.json') as f:
    DATA_INFO   = json.loads(f.read())
    INP_ATTRS   = DATA_INFO['inputs']
    OUT_ATTRS   = DATA_INFO['outputs']
    USE_SCALER  = DATA_INFO['outputs']
    LOOKBACK    = DATA_INFO['lookback']
    LOOKFORWARD = DATA_INFO['lookforward']

for i in range(len(INP_ATTRS)): INP_ATTRS[i] = INP_ATTRS[i].lower()
for i in range(len(OUT_ATTRS)):  OUT_ATTRS[i]  = OUT_ATTRS[i].lower()

INP_ATTR_SIZE = len(INP_ATTRS)
OUT_ATTR_SIZE = len(OUT_ATTRS)

SCALER = []
for attr in INP_ATTRS:
    p = f'./scalers/{attr}.gz'
    SCALER.append(joblib.load(p) if USE_SCALER and os.path.exists(p) else None)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ENCODER_IN_DIM = 5 + INP_ATTR_SIZE * 2
DECODER_TF_DIM = 5

FORECAST_DATA = read_forecast_data('./data/forecast_hr.csv', device=DEVICE, use_scaler=USE_SCALER)
FORECAST_DIM  = FORECAST_DATA.shape[1] if USE_FORECAST else None

loss_fn = LossFN(LOSS_TYPE, SCALER, grad_weight=LOSS_GRAD_WEIGHT)

stn_match_file = open('model_data_1/station_match_1.json')
station_match  = json.loads(stn_match_file.read())
stn_match_file.close()

#####################################################################################
# Data — loaded once, reused by every trial
#####################################################################################
_train_dataset = None
_test_dataset  = None

# best validation loss seen across ALL trials/epochs — gates saving the best model
BEST_MODEL_PATH = 'models/optuna/best_model.pth'
GLOBAL_BEST_VAL = float('inf')

def get_datasets():
    global _train_dataset, _test_dataset
    if _train_dataset is not None:
        return _train_dataset, _test_dataset

    print('Loading datasets for Optuna (once)...')
    train_ds, test_ds = [], []
    for file in FILES:
        stn_data_general = np.load(f'{DATA_DIR_GEN}/{file}.npz')
        stn_data         = np.load(f'{DATA_DIR}/{file}.npz')
        stn_info         = np.array(station_match[file][STN_MATCH_ATTRIBUTE], dtype=np.float32)
        time_feats, target, target_mask, valid_idx = (
            stn_data['time_feats'], stn_data['target'],
            stn_data['target_mask'], stn_data['valid_idx'])
        series    = stn_data_general['series']
        inp       = stn_data_general['input']
        inp_mask  = stn_data_general['input_mask']
        forecast_idx = stn_data['forecast_idx']

        train_idx, test_idx, train_fcast, test_fcast = train_test_split(
            valid_idx, forecast_idx, test_size=0.3, shuffle=False)

        if len(train_idx) > MAX_TRAIN:
            pick = np.round(np.linspace(0, len(train_idx) - 1, MAX_TRAIN)).astype(int)
            train_idx, train_fcast = train_idx[pick], train_fcast[pick]
        if len(test_idx) > MAX_TEST:
            pick = np.round(np.linspace(0, len(test_idx) - 1, MAX_TEST)).astype(int)
            test_idx, test_fcast = test_idx[pick], test_fcast[pick]

        kw = dict(DEVICE=DEVICE, forecast=FORECAST_DATA,
                  use_wavelet=USE_WAVELET, wavelet_level=WAVELET_LVL,
                  inp=inp, inp_mask=inp_mask)
        train_ds.append(StationDataset(series, time_feats, target, target_mask,
                                       train_idx, stn_info, LOOKBACK, LOOKFORWARD,
                                       forecast_idx=train_fcast, **kw))
        test_ds.append( StationDataset(series, time_feats, target, target_mask,
                                       test_idx,  stn_info, LOOKBACK, LOOKFORWARD,
                                       forecast_idx=test_fcast,  **kw))
        print(f'  {file}: {len(train_idx)} train / {len(test_idx)} test')

    _train_dataset = data.ConcatDataset(train_ds)
    _test_dataset  = data.ConcatDataset(test_ds)
    print(f'Total: {len(_train_dataset)} train / {len(_test_dataset)} test')
    return _train_dataset, _test_dataset


#####################################################################################
# Optuna objective
#####################################################################################
def objective(trial: optuna.Trial) -> float:
    # dropout      = trial.suggest_float('dropout',      0.1,  0.6)
    # lr           = trial.suggest_float('lr',           1e-4, 8e-4, log=True)
    # weight_decay = trial.suggest_float('weight_decay', 1e-3, 5e-2, log=True)

    # hidden_dim   = trial.suggest_categorical('hidden_dim', [128, 256, 512])
    # loss_type    = trial.suggest_categorical('loss_type', ['mae', 'mse', 'huber'])
    hidden_dim   = 512
    loss_type    = 'mae'

    dropout      = trial.suggest_categorical('dropout', [0.3, 0.4, 0.5])
    weight_decay = trial.suggest_categorical('weight_decay', [0.01, 0.001])
    lr           = 0.00063

    trial_loss_fn = LossFN(loss_type, SCALER, grad_weight=LOSS_GRAD_WEIGHT)
    # always evaluate with MAE so Optuna compares trials on the same scale
    eval_loss_fn  = LossFN('mae',      SCALER, grad_weight=0)

    train_ds, test_ds = get_datasets()
    train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = data.DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    model = seq2seq_cnn_model(
        ENCODER_IN_DIM, DECODER_TF_DIM, OUT_ATTR_SIZE,
        hidden_dim=hidden_dim, dropout=dropout, forecast_dim=FORECAST_DIM,
    ).to(DEVICE)
    # no torch.compile — compilation overhead per trial is not worthwhile
    # model = torch.compile(model)

    optimizer    = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler    = optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=lr, total_steps=N_OPTUNA_EPOCHS,
    #     pct_start=0.2, div_factor=10, final_div_factor=100, anneal_strategy='cos',
    # )
    scheduler    = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_OPTUNA_EPOCHS, eta_min=lr * 1e-2,
    )
    grad_scaler  = torch.GradScaler(device=DEVICE)

    global GLOBAL_BEST_VAL
    infos = {
        'model_type': MODEL_TYPE, 'inputs': INP_ATTRS, 'outputs': OUT_ATTRS,
        'encoder_in_dim': ENCODER_IN_DIM, 'decoder_tf_dim': DECODER_TF_DIM,
        'use_scaler': USE_SCALER, 'use_wavelet': USE_WAVELET, 'wavelet_lvl': WAVELET_LVL,
        'lookback': LOOKBACK, 'lookforward': LOOKFORWARD, 'forecast_dim': FORECAST_DIM,
        'stn_match_attr': STN_MATCH_ATTRIBUTE,
        'hidden_dim': hidden_dim, 'dropout': dropout, 'lr': lr,
        'weight_decay': weight_decay, 'loss_type': loss_type,
        'optuna_trial': trial.number,
    }

    best_val = float('inf')
    for epoch in range(N_OPTUNA_EPOCHS):
        train_one_epoch_scaler_v1(
            model, train_loader, optimizer, trial_loss_fn, grad_scaler,
            clip_grad=1.0, tf_ratio=0.0, epoch=epoch)
        val = evaluate_v1(model, val_loader, eval_loss_fn, epoch=epoch)

        # save the single best model across all trials, at its best epoch
        if val < GLOBAL_BEST_VAL:
            GLOBAL_BEST_VAL = val
            save_training_state(BEST_MODEL_PATH, model, optimizer, scheduler,
                                grad_scaler, epoch, 0, {'val': val}, infos)
            print(f'  >> new global best {val:.6f} (trial {trial.number}, epoch {epoch}) saved')

        trial.report(val, epoch)
        if trial.should_prune():
            del model
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()

        scheduler.step()
        best_val = min(best_val, val)

    del model
    torch.cuda.empty_cache()
    return best_val


#####################################################################################
if __name__ == '__main__':
    os.makedirs('models/optuna/', exist_ok=True)

    # sampler = optuna.samplers.TPESampler(seed=SEED)
    search_space = {'dropout': [0.3, 0.4, 0.5], 'weight_decay': [0.01, 0.001]}
    sampler = optuna.samplers.GridSampler(search_space)

    # pruner  = optuna.pruners.HyperbandPruner(
    #     min_resource=4, max_resource=N_OPTUNA_EPOCHS, reduction_factor=2)
    pruner  = optuna.pruners.NopPruner()
    study   = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
        storage='sqlite:///models/optuna/study.db',
        study_name='cnn_lstm_hparam_WSpd',
        load_if_exists=True,   # resume if interrupted
    )

    # on resume, seed the global best so we don't overwrite an already-better saved model
    if os.path.exists(BEST_MODEL_PATH):
        completed = [t.value for t in study.trials if t.value is not None]
        if completed:
            GLOBAL_BEST_VAL = min(completed)
            print(f'Resuming: GLOBAL_BEST_VAL seeded to {GLOBAL_BEST_VAL:.6f}')

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print('\n=== Best trial ===')
    t = study.best_trial
    print(f'  Value : {t.value:.6f}')
    for k, v in t.params.items():
        print(f'  {k}: {v}')

    best = {
        'trial_number': t.number,
        'value':        t.value,
        'params':       t.params,
        'n_trials':     len(study.trials),
        'study_name':   study.study_name,
    }
    out_path = 'models/optuna/best_trial.json'
    with open(out_path, 'w') as f:
        json.dump(best, f, indent=2)
    print(f'Saved best trial to {out_path}')
