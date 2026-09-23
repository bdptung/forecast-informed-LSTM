import json
import math
import random
import time
import warnings
import numpy as np
import pandas as pd
import os

import torch
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt

import joblib

from sklearn.model_selection import train_test_split

from utils.reproducibility import seed_everything, tf_func, clip_grad_func
from utils.loss_fn import LossFN, LossFN_winddir
from utils.create_dataset_v1 import StationDataset, make_input_data
from utils.models import (AttrSeq2SeqLSTM as seq2seq_model,
                          AttrSeq2SeqLSTM_v1 as seq2seq_model_v1,
                          AttrSeq2SeqAttnLSTM as seq2seq_attn_model,
                          AttrSeq2SeqBiLSTM as seq2seq_bilstm_model,
                          AttrSeq2SeqCNNLSTM as seq2seq_cnn_model,
                          AttrSeq2SeqCNNAttnLSTM as seq2seq_cnn_attn_model,
                          AttrLSTM as norm_model,
                          AttrLSTMv1 as norm_model_v1,
                          train_one_epoch,
                          evaluate,
                          evaluate_v1,
                          train_one_epoch_scaler,
                          train_one_epoch_scaler_v1,
                          evaluate_scaler
                          )
from utils.utils import EarlyStopper, load_training_state, save_training_state, read_forecast_data
from lstm1__const__ import TRAIN_FILES
import random

SEED = 42     # seed 1
SEED = 1958   # seed 2
SEED = 2662   # seed 3
# SEED = random.randint(2000, 3000)
seed_everything(SEED, False)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def run(DATA_DIR, MODEL_TYPE, USE_FORECAST):
    #####################################################################################
    #####################################################################################
    #####################################################################################
    if not DATA_DIR.endswith('/'): DATA_DIR += '/'
    # DATA_DIR_GEN = 'model_data_preprocessed_nosol/_general/'
    DATA_DIR_GEN = DATA_DIR

    BATCH_SIZE = 256
    HIDDEN_DIM = 512
    NUM_EPOCHS = 50
    EARLY_STOP_WARMUP = 0   # burn-in: early stopping ignores these first epochs. Keep well below the
                            # epoch the model peaks at (~4) or the stopper never sees the true best.
    # BASE_LR = 0.0007810290056285587 #4e-4
    # BASE_LR = 0.00063
    BASE_LR = 0.00005
    DROPOUT = 0.2 ### Temperature & RH
    if 'Wspd' in DATA_DIR or 'WDir' in DATA_DIR:
        DROPOUT = 0.5 ### Wind Spd a& Wind Direction
    WEIGHT_DECAY = 0.001

    # Train/test split by date instead of a per-station ratio: train = windows whose
    # forecast target is before SPLIT_DATE, test = target on/after it.
    SPLIT_DATE = np.datetime64('2025-01-01')
    SPLIT_YEAR = int(SPLIT_DATE.astype('datetime64[Y]').astype(int))  # years since 1970; exact for a Jan-1 cutoff

    SCHEDULER = 'ReduceLROnPlateau' # 'OneCycleLR' 'CosineAnnealingWarmRestarts'
    LOSS_TYPE = 'mae'  # 'mae' | 'mse' | 'huber'
    # if LOSS_TYPE == 'mse':
    #     BASE_LR = BASE_LR ** 0.5
    # Huber transition point per attribute, in the SCALED units the loss sees (~ the model's
    # typical error). Only used when LOSS_TYPE == 'huber'; below delta it's MSE-like, above it
    # MAE-like/robust. Starting points from current per-attribute MAE - tune per run.
    LOSS_DELTA_MAP = {
        'temperature pt100': 0.03,
        'relative humidity':  0.06,
        'wind speed':         0.015,
        'sin_dir':            0.5,   # wind dir: threshold is on angular error Δθ (radians)
        'cos_dir':            0.5,
    }
    STN_MATCH_ATTRIBUTE = 'lat_long_h_scaled'
    LOSS_GRAD_WEIGHT = 0
    # USE_FORECAST = False
    USE_WAVELET  = False
    INCLUDE_YEAR = False
    WAVELET_LVL  = 4
    BATCHNORM = False
    # CNN experiments (toggle here):
    #   current 3-layer pooling : cnn_channels=(64,128,256), pool_size=2, dilation=1
    #   2-layer pooling         : cnn_channels=(64,128),     pool_size=2, dilation=1
    #   dilated, no pooling     : cnn_channels=(64,128),     pool_size=1, dilation=2
    CNN_CONFIG = [[64, 128], 2, 1]

    # model types: norm (predict all future steps at once), seq2seq (predict one step at a time, feeding predictions back in) 
    # MODEL_TYPE = 'plain_lstm'
    # MODEL_TYPE = 'seq2seq_v1'
    # MODEL_TYPE = 'seq2seq_attn'
    # MODEL_TYPE = 'seq2seq_bilstm'
    # MODEL_TYPE = 'seq2seq_cnn'

    DAY = f'{SEED}_{LOSS_TYPE}_{MODEL_TYPE}{'_wd' if USE_WAVELET else ''}{'_fc' if USE_FORECAST else ''}'
    # DAY = f'{LOSS_TYPE}_{MODEL_TYPE}{'_wd' if USE_WAVELET else ''}{'_fc' if USE_FORECAST else ''}'


    FILE_SUFFIX = '_hr_avg.csv'
    FILES = TRAIN_FILES
    # FILES = [
    #     '21004880', '21024003', '21026652', 
    #     # '21026653', # Pinnacle@ Duxton_Ground Along Road #5
    #     '21027656', '21027657', '21027658', 
    #     # '21027659', # Pinnacle@ Duxton_Everton Lamppost #8
    #     '21027660', '21027661', 
    #     '21027662', '21027663', '21027665', '21040276', '21040277', 
    #     '21040278', '21040279', '21040280', '23001059', 
    #     # '23001060', # Fuhua School mid-lvl big podium #4
    #     '23003762', '23003763', '23003764', '23003765', '23003766', 

    #     # 'S06', 
    #     'S100', 'S102', 'S104', 'S106', 
    #     # 'S107', # East Coast Parkway
    #     'S108', 'S109', 
    #     'S111', 'S115', 'S116', 'S117', 
    #     # 'S121', # Old Choa Chu Kang Road
    #     # 'S122', #### no valid data
    #     'S24', 
    #     'S43', 'S44', 'S60', 
    #     # 'S50', # Clementi Road

    #     # obsoletes: 'S24B', 'S96', 'S97'
    # ]


    # LOOKBACK_DAYS = 60
    # # LOOKBACK_DAYS = 30
    # INTERVAL = 60 * 60
    # LOOKBACK_TIME = 60 * 60 * 24 * LOOKBACK_DAYS
    # LOOKBACK = round(LOOKBACK_TIME / INTERVAL)

    # LOOKFORWARD_TIME = 60 * 60 * 24 * 7
    # LOOKFORWARD = round(LOOKFORWARD_TIME / INTERVAL)



    ATTR_MATCH = {
        'temperature pt100': 'Temp',
        'relative humidity': 'RH',
        'wind speed': 'WSpd',
        'sin_wind': 'Wind',
        'cos_wind': 'Wind',
        'sin_dir': 'WDir',
        'cos_dir': 'WDir',
        'solar radiation': 'Sol',
        'pm1.0': 'pm1',
        'pm2.5': 'pm2p5',
        'pm10': 'pm10'
    }

    # BATCH_SIZE, STEPS_PER_EPOCH = 512, 954 #512

    # BATCH_SIZE_EFF = 256
    # MAX_BOUND = None #300000

    # USE_SCALER = True
    # # INP_ATTRS = ['Temperature PT100', 'Relative Humidity', 'Wind speed', 'sin_wind', 'cos_wind', 'Solar radiation', 'PM1.0', 'PM2.5', 'PM10']
    # INP_ATTRS = ['sin_wind', 'cos_wind', 'Solar radiation', 'Temperature PT100', 'Relative Humidity', 'Wind speed']
    # INP_ATTRS = ['sin_wind', 'cos_wind', 'Wind speed', 'Solar radiation', 'Relative Humidity', 'Temperature PT100']
    # # INP_ATTRS = ['Wind speed', 'sin_wind', 'cos_wind', 'Solar radiation', 'Temperature PT100', 'Relative Humidity']
    # # INP_ATTRS = ['Wind speed', 'sin_wind', 'cos_wind', 'Temperature PT100', 'Relative Humidity']

    # # OUT_ATTRS = ['Temperature PT100', 'Relative Humidity']
    # # OUT_ATTRS = ['Temperature PT100', 'Relative Humidity', 'Wind speed', 'sin_wind', 'cos_wind', 'Solar radiation']
    # # OUT_ATTRS = ['Temperature PT100', 'Relative Humidity', 'Wind speed']
    # OUT_ATTRS = ['Relative Humidity']
    # OUT_ATTRS = ['Temperature PT100']

    # for i in range(len(INP_ATTRS)):
    #     INP_ATTRS[i] = INP_ATTRS[i].lower()
    # for i in range(len(OUT_ATTRS)):
    #     OUT_ATTRS[i] = OUT_ATTRS[i].lower()

    with open(DATA_DIR + '_info.json') as f:
        DATA_INFO = json.loads(f.read())
        INP_ATTRS = DATA_INFO['inputs']
        OUT_ATTRS = DATA_INFO['outputs']
        USE_SCALER = DATA_INFO['outputs']
        LOOKBACK = DATA_INFO['lookback']
        LOOKFORWARD = DATA_INFO['lookforward']
        LOOKBACK_DAYS = int(LOOKBACK / 24)

    if len(OUT_ATTRS) <= 2:
        DAY = ATTR_MATCH[OUT_ATTRS[0]] + '_' + DAY

    for i in range(len(INP_ATTRS)):
        INP_ATTRS[i] = INP_ATTRS[i].lower()
    for i in range(len(OUT_ATTRS)):
        OUT_ATTRS[i] = OUT_ATTRS[i].lower()

    print('INP_ATTRS', INP_ATTRS)
    print('OUT_ATTRS', OUT_ATTRS)


    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################


    INP_ATTR_SIZE = len(INP_ATTRS)
    OUT_ATTR_SIZE = len(OUT_ATTRS)
    ATTRS_TRIM = [ATTR.replace(' ', '') for ATTR in INP_ATTRS]

    SCALER = []
    for attr in INP_ATTRS:
        p = f"./scalers/{attr}.gz"
        if USE_SCALER and os.path.exists(p):
            SCALER.append(joblib.load(p))
            print(f'scaler file - OK: ./scalers/{attr}.gz')
        elif not USE_SCALER:
            SCALER.append(None)
            print(f'scaler file - not used per config')
        else:
            SCALER.append(None)
            print(f'scaler file - not exist: ./scalers/{attr}.gz')

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ENCODER_IN_DIM = 8 + 5 + INP_ATTR_SIZE * 2  # station embedding (ID + location) + input time features + past attrs observed status (e.g., wind speed, humidity) + past attrs data (e.g., temp)
    # DECODER_TF_DIM = 8 + 5                      # station embedding (ID + location) + future time features
    ENCODER_IN_DIM = 5 + INP_ATTR_SIZE * 2  # input time features + past attrs observed data (e.g., wind speed, humidity, temp) + past attrs mask (1 if observed, 0 if not observed, for each attr)
    DECODER_TF_DIM = 5                      # future time features

    FORECAST_DATA = read_forecast_data('./data/forecast_hr.csv', device=DEVICE, use_scaler=USE_SCALER)
    # FORECAST_DATA: (T, n_days, day_dim) — forecast is grouped per day for per-step alignment
    FORECAST_DAYS = FORECAST_DATA.shape[1]   # number of forecast days (e.g. 4)
    FORECAST_DIM = FORECAST_DATA.shape[2]    # per-day feature dim (e.g. 10)
    if not USE_FORECAST:
        FORECAST_DIM = None

    OTHER_INFOS = {
        "model_type": MODEL_TYPE,
        'inputs': INP_ATTRS,
        'outputs': OUT_ATTRS,
        'encoder_in_dim': ENCODER_IN_DIM,
        'decoder_tf_dim': DECODER_TF_DIM,
        'use_scaler': USE_SCALER,
        'use_wavelet': USE_WAVELET,
        'include_year': INCLUDE_YEAR,
        'wavelet_lvl': WAVELET_LVL,
        'lookback': LOOKBACK,
        'lookforward': LOOKFORWARD,
        'hidden_dim': HIDDEN_DIM,
        'forecast_dim': FORECAST_DIM,
        'stn_match_attr': STN_MATCH_ATTRIBUTE,
        'cnn_config': CNN_CONFIG,
        'dropout': DROPOUT,
        'batchnorm': BATCHNORM,
        'learning_rate': []
    }
    # print(json.dumps(OTHER_INFOS, indent=2))

    LOSS_DELTA = LOSS_DELTA_MAP.get(OUT_ATTRS[0], 1.0)  # per-attribute Huber transition (OUT_ATTRS is lowercased above)
    loss_fn = LossFN(LOSS_TYPE, SCALER, grad_weight=LOSS_GRAD_WEIGHT, delta=LOSS_DELTA)
    if 'sin_dir' in OUT_ATTRS and 'cos_dir' in OUT_ATTRS:
        loss_fn = LossFN_winddir(LOSS_TYPE, SCALER, grad_weight=LOSS_GRAD_WEIGHT, delta=LOSS_DELTA)
    print(f'LOSS_TYPE={LOSS_TYPE}  LOSS_DELTA={LOSS_DELTA}')

    stn_match_file = open('model_data_1/station_match_1.json')
    station_match = json.loads(stn_match_file.read())
    stn_match_file.close()


    def trainModel( model, optimizer, scheduler, grad_scaler,
                    station_match, last_loss, NUM_EPOCHS, CURRENT_EPOCH, DEVICE,
                    base_lr=4e-4, warmup_epochs=0):
        print('STARTING')
        # Early stopping on validation loss. patience (5) > scheduler patience (2) so the
        # ReduceLROnPlateau LR drop gets a chance to help before we conclude training is done.
        stopper = EarlyStopper(patience=5, min_rel_delta=0.001)  # need >0.1% rel. val improvement to reset
        prev_test = last_loss.get('test', {}).get('mean')
        if prev_test:
            # stopper.best = min(prev_test)   # carry best across resumes so patience doesn't reset each run
            for i, va in enumerate(prev_test):
                print(i, va)
                should_stop = stopper.step(va) if i >= EARLY_STOP_WARMUP else False
                if should_stop:
                    print('!!!RUN ALREADY FINISHED -- STOPPING!!!')
                    return
    
        MAX_TRAIN = 12000
        MAX_TEST  = 4000
        full_test_dataset = None
        test_datasets = []
        for _ in range(NUM_EPOCHS - CURRENT_EPOCH):
            if CURRENT_EPOCH < warmup_epochs:
                warmup_lr = base_lr * (CURRENT_EPOCH + 1) / warmup_epochs
                for pg in optimizer.param_groups:
                    pg['lr'] = warmup_lr

            train_datasets = []

            for file in FILES:
                stn_data_general = np.load(f'{DATA_DIR_GEN}/{file}.npz')
                stn_data = np.load(f'{DATA_DIR}/{file}.npz')
                stn_info = np.array(station_match[file][STN_MATCH_ATTRIBUTE], dtype=np.float32)
                time_feats, target, target_mask, valid_idx = stn_data['time_feats'], stn_data['target'], stn_data['target_mask'], stn_data['valid_idx']
                series = stn_data_general['series']
                inp = stn_data_general['input']
                inp_mask = stn_data_general['input_mask']
                forecast_idx = stn_data['forecast_idx']
                # Date-based split on the forecast target period: TEST = windows whose 168h target
                # starts on/after SPLIT_DATE; TRAIN = windows whose target ends before SPLIT_DATE.
                # Windows straddling the boundary are dropped so no training target overlaps the test
                # period. Exact for a Jan-1 cutoff: time_feats[:, 4] is the absolute year (years since
                # 1970, base_year=0), so the year comparison IS the date comparison.
                year_of = time_feats[:, 4]
                tgt_start_year = year_of[valid_idx + LOOKBACK]
                tgt_end_year   = year_of[valid_idx + LOOKBACK + LOOKFORWARD - 1]
                is_test  = tgt_start_year >= SPLIT_YEAR
                is_train = tgt_end_year   <  SPLIT_YEAR
                train_idx, train_forecast = valid_idx[is_train], forecast_idx[is_train]
                test_idx,  test_forecast  = valid_idx[is_test],  forecast_idx[is_test]
                # year_norm (time_feats[:, 4]) is the ABSOLUTE year, so with a date-based split it is
                # <=54 in train but 55 in test - a value the model never sees while training. Left in,
                # the model keys on it to fit year-specific quirks and then extrapolates at test time
                # (test loss rises as train falls). Zero it once the split is decided; seasonality still
                # lives in doy_sin/doy_cos. series[:, :5] IS time_feats, so both need zeroing.
                if not INCLUDE_YEAR:
                    time_feats[:, 4] = 0.0
                    series[:, 4] = 0.0
                # Evenly-spaced subsample (re-drawn each epoch, since this runs inside the epoch loop),
                # so the model sees a different slice of each station's windows every epoch.
                # The whole grid is shifted right by HOUR_SHIFT positions per epoch: train_idx is
                # chronological and (bar missing-data gaps) hourly, so +HOUR_SHIFT positions ~= a
                # +HOUR_SHIFT-hour shift of every kept window's start time. Headroom of
                # HOUR_SHIFT*NUM_EPOCHS windows is reserved at the tail so late epochs never run
                # off the end (no clipping / duplicate picks).
                HOUR_SHIFT = 5
                max_off = HOUR_SHIFT * NUM_EPOCHS
                if len(train_idx) > (MAX_TRAIN + max_off):
                    step   = (len(train_idx) - max_off) / MAX_TRAIN          # spacing between kept windows
                    offset = CURRENT_EPOCH * HOUR_SHIFT                      # cumulative 5h/epoch phase shift
                    pick   = np.round(offset + np.arange(MAX_TRAIN) * step).astype(int)
                    train_idx = train_idx[pick]
                    train_forecast = train_forecast[pick]
                train_datasets.append(StationDataset(
                    series, time_feats, target, target_mask, train_idx, stn_info, 
                    LOOKBACK, LOOKFORWARD, forecast=FORECAST_DATA, forecast_idx=train_forecast,
                    use_wavelet=USE_WAVELET, wavelet_level=WAVELET_LVL, inp=inp, inp_mask=inp_mask))
                if full_test_dataset is None:
                    if len(test_idx) > MAX_TEST:
                        pick = np.round(np.linspace(0, len(test_idx) - 1, MAX_TEST)).astype(int)
                        test_idx = test_idx[pick]
                        test_forecast = test_forecast[pick]
                    test_datasets.append(StationDataset(
                        series, time_feats, target, target_mask, test_idx, stn_info,
                        LOOKBACK, LOOKFORWARD, forecast=FORECAST_DATA, forecast_idx=test_forecast,
                        use_wavelet=USE_WAVELET, wavelet_level=WAVELET_LVL, inp=inp, inp_mask=inp_mask))
                print(f'Loaded data for file: {file}, train-test size: {len(train_idx)}-{len(test_idx)}')
            full_train_dataset = data.ConcatDataset(train_datasets)
            if full_test_dataset is None:
                full_test_dataset = data.ConcatDataset(test_datasets)
            print(len(full_train_dataset), len(full_test_dataset))


            OTHER_INFOS['learning_rate'].append(optimizer.param_groups[0]['lr'])
            loader = data.DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
            print('training', DAY)
            tr = train_one_epoch_scaler_v1(model, loader, optimizer, loss_fn, grad_scaler, clip_grad=clip_grad_func(CURRENT_EPOCH), tf_ratio=0, epoch=CURRENT_EPOCH)
            val_dl = data.DataLoader(full_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            va = evaluate_v1(model, val_dl, loss_fn, epoch=CURRENT_EPOCH)
            if 'mean' not in last_loss['train'] or 'mean' not in last_loss['test']:
                last_loss['train']['mean'] = []
                last_loss['test']['mean'] = []
            last_loss['train']['mean'].append(tr)
            last_loss['test']['mean'].append(va)
            # Burn-in: ignore the noisy first few epochs so early stopping doesn't anchor its
            # 'best' to a fluky early value or fire before the model has settled.
            should_stop = stopper.step(va) if CURRENT_EPOCH >= EARLY_STOP_WARMUP else False

            if CURRENT_EPOCH >= warmup_epochs:
                if SCHEDULER == 'ReduceLROnPlateau':
                    scheduler.step(last_loss['test']['mean'][-1]) 
                else:
                    scheduler.step()


            f = f"models/LSTM_{DAY}/model.epoch_{str(CURRENT_EPOCH)}.pth"
            save_training_state(f, model, optimizer, scheduler, grad_scaler, CURRENT_EPOCH, 0, last_loss, OTHER_INFOS)
            CURRENT_EPOCH += 1
            print(last_loss['test']['mean'])
            if should_stop:
                print(f'>>> Early stop at epoch {CURRENT_EPOCH - 1}: no >{stopper.min_rel_delta:.1%} val '
                      f'improvement for {stopper.patience} epochs (best val {stopper.best:.5f})')
                break

        print('!!!DONE!!!')


    def loadModel(model, optimizer, scheduler, grad_scaler, NUM_EPOCHS):
        filename = None
        current_epoch = 0
        MODEL_LOADED = None
        for i in range(0, NUM_EPOCHS):
            f = f"models/LSTM_{DAY}/model.epoch_{str(i)}.pth"
            if os.path.exists(f):
                filename = f
                current_epoch = i + 1
        if filename is not None:
            print('loading model:', filename)
            MODEL_LOADED = load_training_state(filename, model, optimizer, scheduler, grad_scaler)
            print('Model loaded:', filename)
        return current_epoch, MODEL_LOADED

    def run0():
        MODEL_LOADED = None
        CURRENT_EPOCH = 0

        # model = norm_model(ENCODER_IN_DIM, OUT_ATTR_SIZE, hidden_dim=HIDDEN_DIM)
        model = None
        if MODEL_TYPE == 'seq2seq':
            model = seq2seq_model(ENCODER_IN_DIM, DECODER_TF_DIM, OUT_ATTR_SIZE, hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
        elif MODEL_TYPE == 'seq2seq_v1':
            model = seq2seq_model_v1(ENCODER_IN_DIM, DECODER_TF_DIM, OUT_ATTR_SIZE, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, forecast_dim=FORECAST_DIM, forecast_days=FORECAST_DAYS)
        elif MODEL_TYPE == 'seq2seq_attn':
            model = seq2seq_attn_model(ENCODER_IN_DIM, DECODER_TF_DIM, OUT_ATTR_SIZE, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, forecast_dim=FORECAST_DIM, forecast_days=FORECAST_DAYS)
        elif MODEL_TYPE == 'seq2seq_bilstm':
            model = seq2seq_bilstm_model(ENCODER_IN_DIM, DECODER_TF_DIM, OUT_ATTR_SIZE, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, forecast_dim=FORECAST_DIM, forecast_days=FORECAST_DAYS)
        elif MODEL_TYPE == 'seq2seq_cnn':
            model = seq2seq_cnn_model(
                ENCODER_IN_DIM, DECODER_TF_DIM, OUT_ATTR_SIZE,
                hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
                forecast_dim=FORECAST_DIM, forecast_days=FORECAST_DAYS,
                cnn_channels=CNN_CONFIG[0], pool_size=CNN_CONFIG[1], dilation=CNN_CONFIG[2],
            )
        elif MODEL_TYPE == 'seq2seq_cnn_attn':
            model = seq2seq_cnn_attn_model(
                ENCODER_IN_DIM, DECODER_TF_DIM, OUT_ATTR_SIZE,
                hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
                forecast_dim=FORECAST_DIM, forecast_days=FORECAST_DAYS,
                cnn_channels=CNN_CONFIG[0], pool_size=CNN_CONFIG[1], dilation=CNN_CONFIG[2],
            )
        elif MODEL_TYPE == 'plain_lstm':
            model = norm_model(ENCODER_IN_DIM, DECODER_TF_DIM, OUT_ATTR_SIZE, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, forecast_dim=FORECAST_DIM, forecast_days=FORECAST_DAYS)
        elif MODEL_TYPE == 'v1':
            model = norm_model_v1(ENCODER_IN_DIM, DECODER_TF_DIM, OUT_ATTR_SIZE, hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
        else:
            model = norm_model(ENCODER_IN_DIM, DECODER_TF_DIM, OUT_ATTR_SIZE, hidden_dim=HIDDEN_DIM, dropout=DROPOUT)

        model = model.to(DEVICE)
        model = torch.compile(model)
        WARMUP_EPOCHS = EARLY_STOP_WARMUP
        optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
        # optimizer = optim.Adam(model.parameters(), lr=4e-4)
        if SCHEDULER == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, min_lr=1e-6,
                patience=2, cooldown=2   # converges by ~ep4, so react fast; cooldown avoids noise-driven back-to-back drops
                # patience=10
            )
        elif SCHEDULER == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=15,      # first restart at epoch 15
                T_mult=2,    # subsequent cycles: 15 → 30 → 60 epochs
                eta_min=5e-6
            )
        elif SCHEDULER == 'OneCycleLR':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=BASE_LR,          
                total_steps=NUM_EPOCHS,
                pct_start=0.2,         
                div_factor=10,         
                final_div_factor=100,  
                anneal_strategy='cos',
            )

        grad_scaler = torch.GradScaler(device=DEVICE)

        last_loss = {'train': {},'test': {}}


        print('cuda available:', torch.cuda.is_available(), DEVICE)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        CURRENT_EPOCH, MODEL_LOADED = loadModel(model, optimizer, scheduler, grad_scaler, NUM_EPOCHS)
        if MODEL_LOADED and 'prev_losses' in MODEL_LOADED:
            last_loss = MODEL_LOADED['prev_losses']

        # If scheduler state was incompatible (e.g. switched from ReduceLROnPlateau to CAWR),
        # fast-forward the scheduler to match CURRENT_EPOCH so the LR cycle is in the right position.
        if CURRENT_EPOCH > 0 and scheduler.last_epoch == 0:
            for l in range(CURRENT_EPOCH):
                if SCHEDULER == 'ReduceLROnPlateau':
                    scheduler.step(last_loss['test']['mean'][l])  # ReduceLROnPlateau
                else:
                    scheduler.step()
        print('..........', last_loss)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        trainModel(
        model, optimizer, scheduler, grad_scaler,
        station_match, last_loss, NUM_EPOCHS, CURRENT_EPOCH, DEVICE,
        base_lr=BASE_LR, warmup_epochs=WARMUP_EPOCHS)

    if not os.path.exists(f'models/LSTM_{DAY}/'):
        os.mkdir(f'models/LSTM_{DAY}/')
    run0()

if __name__ == "__main__":
    RUN_TYPES = [
        {'MODEL_TYPE': 'seq2seq_v1', 'USE_FORECAST': True},
        {'MODEL_TYPE': 'seq2seq_v1', 'USE_FORECAST': False},
        {'MODEL_TYPE': 'plain_lstm', 'USE_FORECAST': True},
        {'MODEL_TYPE': 'plain_lstm', 'USE_FORECAST': False},
        {'MODEL_TYPE': 'seq2seq_attn', 'USE_FORECAST': True},
        {'MODEL_TYPE': 'seq2seq_cnn', 'USE_FORECAST': True},
        {'MODEL_TYPE': 'seq2seq_cnn_attn', 'USE_FORECAST': True},
    ]
    RUN_DIRS = [
        'model_data_preprocessed_nosol/Temp_60',
        'model_data_preprocessed_nosol/RH_60',
        'model_data_preprocessed_nosol/WSpd_60',
        'model_data_preprocessed_nosol/WDir_60'
    ]
    for runtype in RUN_TYPES:
        for datadir in RUN_DIRS:
            run(datadir, runtype['MODEL_TYPE'], runtype['USE_FORECAST'])
