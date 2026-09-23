import numpy as np

from sklearn.model_selection import train_test_split
import torch


# ---------- Time feature builder: minute-of-day, day-of-year, year ----------
def make_time_features_v2(station_id_match: np.ndarray, timestamps: np.ndarray, base_year: int = 0) -> np.ndarray:
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
    stnPad = np.tile(station_id_match, (feats.shape[0], 1))
    feats = np.hstack([feats, stnPad])
    # feats = np.pad(feats, ((0, 0), (1, 0)), 'constant', constant_values=station_id_match)
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
    time_features = make_time_features_v2(station_id_match, timestamp_col)

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
    time_features = make_time_features_v2(station_id_match, timestamp_col)

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

