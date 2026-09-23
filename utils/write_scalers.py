import math
import numpy as np
import pandas as pd


from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import joblib

DATA_DIR = './model_data_1/'
FILE_SUFFIX = '_hr_avg.csv'


SCALER = {}
FILES = [
    '21004880', '21024003', '21026652', '21026653', '21027656', 
    '21027657', '21027658', '21027659', '21027660', '21027661', 
    '21027662', '21027663', '21027665', '21040276', '21040277', 
    '21040278', '21040279', '21040280', '23001059', '23001060', 
    '23003762', '23003763', '23003764', '23003765', '23003766', 
    'S06', 'S100', 'S102', 'S104', 'S106', 'S107', 'S108', 'S109', 
    'S111', 'S115', 'S116', 'S117', 'S121', 'S122', 'S24', 
    'S43', 'S44', 'S50', 'S60', 
    # 'S24B', 'S96', 'S97'
    ] 
ATTRIBUTES = [
    'wind speed', 'temperature pt100', 'relative humidity', 
    'wind direction', 
    'cos_wind', 'sin_wind'
    # , 'cos_dir', 'sin_dir'
    , 'solar radiation', 'pm1', 'pm2p5', 'pm10'
]
for attr in ATTRIBUTES:
    attrVals = np.array([], dtype='float32')
    for file in FILES:
        dataset = pd.read_csv(f'{DATA_DIR}{file}{FILE_SUFFIX}')
        if attr not in dataset: continue
        attrVals = np.concatenate((attrVals, dataset[attr].values.astype('float32').flatten()), axis=0) 
        # # Ensure the dataset is sorted by timestamp
    # if attr in ['...', ]:
    #     # RobustScaler has no feature_range (output is unbounded by design):
    #     # centers on the median, scales by the IQR (quantile_range)
    #     scaler = RobustScaler(quantile_range=(25.0, 75.0))
    # else:
    #     scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(attrVals.reshape(-1, 1))
    joblib.dump(scaler, f"./scalers/{attr}.gz") 
    print(attr, attrVals.shape)