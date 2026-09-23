import numpy as np
import torch.utils.data as data

from sklearn.model_selection import train_test_split
import torch
import ptwt, pywt
import time

# ---------- Time feature builder: minute-of-day, day-of-year, year ----------
def make_time_features_v2(timestamps: np.ndarray, base_year: int = 0) -> np.ndarray:
    """
    timestamps: np.array convertible to numpy datetime64[m]
    Returns features: [minute_sin, minute_cos, doy_sin, doy_cos, year_norm]
    """
    ts = np.array(timestamps).astype('datetime64[m]')

    # Minute of day (0–1439)
    minutes = ((ts - ts.astype('datetime64[D]')) / np.timedelta64(1, 'm')).astype(int)
    minute_sin = np.sin(2*np.pi*minutes/1440)
    minute_cos = np.cos(2*np.pi*minutes/1440)

    # Day of year (0–365)
    days_into_year = (ts.astype('datetime64[D]') - ts.astype('datetime64[Y]')) / np.timedelta64(1, 'D')
    doy = days_into_year.astype(int)
    doy_sin = np.sin(2*np.pi*doy/365)
    doy_cos = np.cos(2*np.pi*doy/365)

    # Year as trend
    years = ts.astype('datetime64[Y]').astype(int)
    if base_year is None:
        base_year = years.min()
    year_norm = (years - base_year).astype(float)

    feats = np.stack([minute_sin, minute_cos, doy_sin, doy_cos, year_norm], axis=-1)
    return feats.astype(np.float32)

def nan_check(nanset, size, attrs, tolerance = 0.2):
    check = True
    for i in range(len(attrs)):
        attr = attrs[i]
        nans = nanset[attr].sum()
        if nans == size: continue
        # if nans > 0:
        if (nans / size) < tolerance:
            check = False
            break
    return check

def create_dataset(dataset, scalers, lookback, lookforward, station_id_match, attrs, from_idx, to_idx):
    end_idx = to_idx
    target_end_idx = end_idx + lookback + lookforward
    if target_end_idx > len(dataset):
        target_end_idx = len(dataset)
        end_idx = target_end_idx - lookback - lookforward
        if end_idx < from_idx:
            return None
        
    timestamp_col = dataset['Timestamp'].dt.tz_localize(None)
    time_features = make_time_features_v2(timestamp_col)

    column_data = dataset[attrs]
    column_data_nan = np.logical_not(np.isnan(column_data)).astype(int)
    nans_counts = {}
    stacked_data = []
    # column_data = dataset[[attr]].values.astype('float32')
    for i in range(len(attrs)):
        attr = attrs[i]
        if attr not in dataset.columns:
            continue
        if scalers[i] is not None:
          scaled = scalers[i].transform(column_data[attr].values.astype('float32').reshape(-1, 1)).reshape(-1)
          stacked_data.append(scaled)
        else:
          stacked_data.append(column_data[attr].values.astype('float32'))
        nans_counts[attr] = np.sum(column_data_nan[attr])
    column_data = np.stack(stacked_data, axis=1)
    column_data[np.isnan(column_data)] = 0
    combined_data = np.concatenate((time_features, column_data_nan, column_data), axis=1).astype('float32')
    # X = np.ndarray(shape=(shp_size, lookback, 1), dtype=np.float32)
    # y = np.ndarray(shape=(shp_size, lookback, 1), dtype=np.float32)

    # X = np.ndarray(shape=(shp_size, lookback), dtype=np.float32)
    X = []
    t = []
    m = []
    y = []
    for i in range(from_idx, end_idx):
        if not nan_check(column_data_nan[i:i+lookback], lookback, attrs): continue
        if not nan_check(column_data_nan[i+lookback:i+lookback+lookforward], lookforward, attrs): continue
        feature = combined_data[i:i+lookback]
        target_info = time_features[i+lookback:i+lookback+lookforward]
        target_mask = column_data_nan[i+lookback:i+lookback+lookforward]
        target_val = column_data[i+lookback:i+lookback+lookforward]

        X.append(feature)
        t.append(target_info)
        m.append(target_mask)
        y.append(target_val)
    return np.array(X), np.array(t), np.array(m), np.array(y)

def create_dataset_diff_input(dataset, scalers, lookback, lookforward, station_id_match, input_attrs, output_attrs, from_idx, to_idx):
    end_idx = to_idx
    target_end_idx = end_idx + lookback + lookforward
    if target_end_idx > len(dataset):
        target_end_idx = len(dataset)
        end_idx = target_end_idx - lookback - lookforward
        if end_idx < from_idx:
            return None
    for i in output_attrs:
      if i not in input_attrs:
        raise Exception(f'ERROR: Output attribute {i} not in input attributes list: {input_attrs}')
    timestamp_col = dataset['timestamp'].dt.tz_localize(None)
    time_features = make_time_features_v2(timestamp_col)

    for attr in input_attrs:
        if attr not in dataset: dataset[attr] = np.nan
    input_data = dataset[input_attrs]
    input_data_nan = np.logical_not(np.isnan(input_data)).astype(int)

    target_column_data = dataset[output_attrs]
    target_data_nan = np.logical_not(np.isnan(target_column_data)).astype(int)
    nans_counts = {}
    stacked_inp_data = []
    stacked_tar_data = []
    # column_data = dataset[[attr]].values.astype('float32')
    for i in range(len(input_attrs)):
        attr = input_attrs[i]
        if attr not in dataset.columns:
            continue
        if scalers[i] is not None:
          scaled = scalers[i].transform(input_data[attr].values.astype('float32').reshape(-1, 1)).reshape(-1)
          stacked_inp_data.append(scaled)
          if attr in output_attrs:
            stacked_tar_data.append(scaled)
        else:
          col = input_data[attr].values.astype('float32')
          stacked_inp_data.append(col)
          if attr in output_attrs:
            stacked_tar_data.append(col)
        nans_counts[attr] = np.sum(input_data_nan[attr])
    input_data = np.stack(stacked_inp_data, axis=1)
    input_data[np.isnan(input_data)] = 0
    target_data = np.stack(stacked_tar_data, axis=1)
    target_data[np.isnan(target_data)] = 0
    combined_data = np.concatenate((time_features, input_data_nan, input_data), axis=1).astype('float32')
    # X = np.ndarray(shape=(shp_size, lookback, 1), dtype=np.float32)
    # y = np.ndarray(shape=(shp_size, lookback, 1), dtype=np.float32)

    # X = np.ndarray(shape=(shp_size, lookback), dtype=np.float32)
    X = []
    t = []
    m = []
    y = []
    for i in range(from_idx, end_idx):
        if not nan_check(target_data_nan[i:i+lookback], lookback, output_attrs): continue
        if not nan_check(target_data_nan[i+lookback:i+lookback+lookforward], lookforward, output_attrs): continue
        feature = combined_data[i:i+lookback]
        target_info = time_features[i+lookback:i+lookback+lookforward]
        target_mask = target_data_nan[i+lookback:i+lookback+lookforward]
        target_val = target_data[i+lookback:i+lookback+lookforward]

        X.append(feature)
        t.append(target_info)
        m.append(target_mask)
        y.append(target_val)
    return np.array(X), np.array(t), np.array(m), np.array(y)


def make_input_data(dataset, scalers, lookback, lookforward, station_id_match, attrs, from_idx, to_idx, device='cuda', seed=42, outputAttrs=None):
    data_package = None
    if outputAttrs is not None:
      data_package = create_dataset_diff_input(dataset, scalers, lookback, lookforward, station_id_match, attrs, outputAttrs, from_idx, to_idx)
    else:
      data_package = create_dataset(dataset, scalers, lookback, lookforward, station_id_match, attrs, from_idx, to_idx)

    if data_package is None: return None
    Xset, Tset, Mset, Yset = data_package

    # Xtensor = torch.tensor(Xset, device=device)
    # Ytensor = torch.tensor(Yset, device=device)
    try:
        X_train_0, X_test_0, t_train_0, t_test_0, m_train_0, m_test_0, y_train_0, y_test_0 = train_test_split( Xset, Tset, Mset, Yset, test_size=0.3, shuffle=False)
    except: 
        return None
    X_train = torch.tensor(X_train_0, device=device)
    X_test = torch.tensor(X_test_0, device=device)

    t_train = torch.tensor(t_train_0, device=device)
    t_test = torch.tensor(t_test_0, device=device)

    m_train = torch.tensor(m_train_0, device=device)
    m_test = torch.tensor(m_test_0, device=device)

    y_train = torch.tensor(y_train_0, device=device)
    y_test = torch.tensor(y_test_0, device=device)

    return X_train, X_test, t_train, t_test, m_train, m_test, y_train, y_test


class StationDataset(data.Dataset):
    def __init__(self, series, time_feats, target_data, target_mask, valid_idx, station_info, lookback, lookforward, 
                 DEVICE='cuda', forecast=None, forecast_idx=None,
                 use_wavelet=False, wavelet_name='db4', wavelet_level=6,
                 inp=None, inp_mask=None):
        self.series         = torch.tensor(series, dtype=torch.float32, device=DEVICE)  # (T, enc_dim)
        self.time_feats     = torch.tensor(time_feats, device=DEVICE)   # (T, 5)
        self.target         = torch.tensor(target_data, device=DEVICE)  # (T, out_attrs)
        self.target_mask    = torch.tensor(target_mask, device=DEVICE)
        # Precompute, for each timestep t, the index of the most recent fully-present target hour
        # (<= t): O(T) once, so __getitem__ can O(1)-anchor the decoder to the last PRESENT value
        # instead of scanning the mask every call. -1 before any present hour occurs.
        _m = np.asarray(target_mask)
        _present = (_m > 0).reshape(_m.shape[0], -1).all(axis=1)
        self.last_present = np.maximum.accumulate(np.where(_present, np.arange(_m.shape[0]), -1))
        self.valid_idx      = valid_idx
        self.station_feats  = torch.tensor(station_info, device=DEVICE) # (1, 3) or similar
        self.lookback       = lookback
        self.lookforward    = lookforward
        self.DEVICE         = DEVICE
        self.forecast       = forecast
        self.forecast_idx   = forecast_idx
        self.use_wavelet    = use_wavelet
        if use_wavelet:
            self._wavelet    = pywt.Wavelet(wavelet_name)
            self._wav_level  = wavelet_level
            self._n_bands    = wavelet_level + 1  # cA_L, cD_L, ..., cD_1
            # enc_dim after wavelet: original * n_bands
            self.enc_dim_out = series.shape[1] * self._n_bands

            self._inp = None
            self._inp_mask = None
            if inp is not None and inp_mask is not None:
                self._inp      = inp
                self._inp_mask = torch.tensor(inp_mask, dtype=torch.float32, device=DEVICE)
                self.enc_dim_out = self._inp.shape[1] * self._n_bands + self._inp_mask.shape[1] + self.time_feats.shape[1]
            else:
                raise Exception('input and input_mask are required for use_wavelet=True')
        else:
            self.enc_dim_out = series.shape[1]

    def __len__(self):
        return len(self.valid_idx)

    def _apply_wavelet(self, X):
        x_np = X.T  # (enc_dim_ch, lookback)
        coeffs = pywt.wavedec(x_np, self._wavelet, level=self._wav_level,
                            mode='periodization', axis=1)
        cat = np.concatenate(coeffs, axis=1)  # (enc_dim_ch, lookback) — exact, no pad
        return torch.tensor(cat.T, dtype=torch.float32, device=self.DEVICE)  # (lookback, enc_dim_ch)

    def __getitem__(self, i):
        s = self.valid_idx[i]

        if self.use_wavelet:
            if self._inp is not None and self._inp_mask is not None:
                X0 = self._inp[s : s + self.lookback]
                X0 = self._apply_wavelet(X0)
                m0 = self._inp_mask[s : s + self.lookback]
                t0 = self.time_feats[s : s + self.lookback]
                X = torch.cat((t0, m0, X0), dim=1)
            else:
                raise Exception('input and input_mask are required for use_wavelet=True')
        else:
            X = self.series[s : s + self.lookback]

        t = self.time_feats[s + self.lookback : s + self.lookback + self.lookforward]
        m = self.target_mask[s + self.lookback : s + self.lookback + self.lookforward]
        y = self.target[s + self.lookback : s + self.lookback + self.lookforward]
        # Anchor the decoder with the last PRESENT target value in the lookback (the final hour may
        # be a NaN->0 gap; seeding with 0 ~= the mean biases the first forecast step low). O(1) via
        # the precomputed last-present index; fall back to the final hour if the window is all gaps.
        lp = int(self.last_present[s + self.lookback - 1])
        if lp < s:
            lp = s + self.lookback - 1
        l = self.target[lp: lp + 1]
        forecast = None
        if self.forecast is not None and self.forecast_idx is not None:
           forecast = self.forecast[self.forecast_idx[i]]
        return X, t, m, y, self.station_feats, l, forecast
