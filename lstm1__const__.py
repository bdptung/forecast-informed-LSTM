TRAIN_FILES = [
    ############ train stations ############
    '21004880', '21024003', '21026652', 
    '21027656', '21027657', '21027658', 
    '21027660', '21027661', 
    '21027662', '21027663', 
    # '21027665', ## Singapore General Hospital #4
    '21040276', '21040277', 
    '21040278', '21040279', '21040280', '23001059', 
    '23003762', '23003763', '23003764', '23003765', '23003766', 

    'S100', 'S102', 'S104', 'S106', 
    'S108', 'S109', 
    'S111', 'S115', 'S116', 'S117', 
    'S24', 
    'S43', 'S44', 'S60', 
]
TEST_FILES = [
    ############ test stations #############
    '21027659', # Pinnacle@ Duxton_Everton Lamppost #8
    '23001060', # Fuhua School mid-lvl big podium #4
    '21026653', # Pinnacle@ Duxton_Ground Along Road #5
    '21027665', ## Singapore General Hospital #4
    'S06', 
    'S50', # Clementi Road
    'S107', # East Coast Parkway
    'S121', # Old Choa Chu Kang Road

    # # obsoletes: 'S24B', 'S96', 'S97', 'S122'
] 



DAYS = {
  "LSTM": {
    "Temp": "LSTM_Temp_huber0_plain_lstm",
    "RH": "LSTM_RH_huber0_plain_lstm",
    "WSpd": "LSTM_WSpd_huber0_plain_lstm",
    "WDir": "LSTM_WDir_huber0_plain_lstm"
  },
  "LSTM-FC": {
    "Temp": "LSTM_Temp_huber0_plain_lstm_fc",
    "RH": "LSTM_RH_huber0_plain_lstm_fc",
    "WSpd": "LSTM_WSpd_huber0_plain_lstm_fc",
    "WDir": "LSTM_WDir_huber0_plain_lstm_fc"
  },
  "LSTM-ED": {
    "Temp": "LSTM_Temp_huber0_seq2seq_v1",
    "RH": "LSTM_RH_huber0_seq2seq_v1",
    "WSpd": "LSTM_WSpd_huber0_seq2seq_v1",
    "WDir": "LSTM_WDir_huber0_seq2seq_v1"
  },
  "LSTM-ED-FC": {
    "Temp": "LSTM_Temp_huber0_seq2seq_v1_fc",
    "RH": "LSTM_RH_huber0_seq2seq_v1_fc",
    "WSpd": "LSTM_WSpd_huber0_seq2seq_v1_fc",
    "WDir": "LSTM_WDir_huber0_seq2seq_v1_fc"
  },
  "LSTM-ED-ATTN-FC": {
    "Temp": "LSTM_Temp_huber0_seq2seq_attn_fc",
    "RH": "LSTM_RH_huber0_seq2seq_attn_fc",
    "WSpd": "LSTM_WSpd_huber0_seq2seq_attn_fc",
    "WDir": "LSTM_WDir_huber0_seq2seq_attn_fc"
  },
  "LSTM-ED-CNN-FC": {
    "Temp": "LSTM_Temp_huber0_seq2seq_cnn_fc",
    "RH": "LSTM_RH_huber0_seq2seq_cnn_fc",
    "WSpd": "LSTM_WSpd_huber0_seq2seq_cnn_fc",
    "WDir": "LSTM_WDir_huber0_seq2seq_cnn_fc"
  }
}

DAYS = {
  "LSTM": {
    "Temp": "LSTM_Temp_mae_plain_lstm",
    "RH": "LSTM_RH_mae_plain_lstm",
    "WSpd": "LSTM_WSpd_mae_plain_lstm",
    "WDir": "LSTM_WDir_mae_plain_lstm"
  },
  "LSTM-FC": {
    "Temp": "LSTM_Temp_mae_plain_lstm_fc",
    "RH": "LSTM_RH_mae_plain_lstm_fc",
    "WSpd": "LSTM_WSpd_mae_plain_lstm_fc",
    "WDir": "LSTM_WDir_mae_plain_lstm_fc"
  },
  "LSTM-ED": {
    "Temp": "LSTM_Temp_mae_seq2seq_v1",
    "RH": "LSTM_RH_mae_seq2seq_v1",
    "WSpd": "LSTM_WSpd_mae_seq2seq_v1",
    "WDir": "LSTM_WDir_mae_seq2seq_v1"
  },
  "LSTM-ED-FC": {
    "Temp": "LSTM_Temp_mae_seq2seq_v1_fc",
    "RH": "LSTM_RH_mae_seq2seq_v1_fc",
    "WSpd": "LSTM_WSpd_mae_seq2seq_v1_fc",
    "WDir": "LSTM_WDir_mae_seq2seq_v1_fc"
  },
  "LSTM-ED-ATTN-FC": {
    "Temp": "LSTM_Temp_mae_seq2seq_attn_fc",
    "RH": "LSTM_RH_mae_seq2seq_attn_fc",
    "WSpd": "LSTM_WSpd_mae_seq2seq_attn_fc",
    "WDir": "LSTM_WDir_mae_seq2seq_attn_fc"
  },
  "LSTM-ED-CNN-FC": {
    "Temp": "LSTM_Temp_mae_seq2seq_cnn_fc",
    "RH": "LSTM_RH_mae_seq2seq_cnn_fc",
    "WSpd": "LSTM_WSpd_mae_seq2seq_cnn_fc",
    "WDir": "LSTM_WDir_mae_seq2seq_cnn_fc"
  },
  # "LSTM-ED-CNN-ATTN-FC": {
  #   "Temp": "LSTM_Temp_mae_seq2seq_cnn_attn_fc",
  #   "RH": "LSTM_RH_mae_seq2seq_cnn_attn_fc",
  #   "WSpd": "LSTM_WSpd_mae_seq2seq_cnn_attn_fc",
  #   "WDir": "LSTM_WDir_mae_seq2seq_cnn_attn_fc"
  # }
}



# DAYS = {
#   "LSTM": {
#     "Temp": "LSTM_Temp_1958_mae_plain_lstm",
#     "RH": "LSTM_RH_1958_mae_plain_lstm",
#     "WSpd": "LSTM_WSpd_1958_mae_plain_lstm",
#     "WDir": "LSTM_WDir_1958_mae_plain_lstm"
#   },
#   "LSTM-FC": {
#     "Temp": "LSTM_Temp_1958_mae_plain_lstm_fc",
#     "RH": "LSTM_RH_1958_mae_plain_lstm_fc",
#     "WSpd": "LSTM_WSpd_1958_mae_plain_lstm_fc",
#     "WDir": "LSTM_WDir_1958_mae_plain_lstm_fc"
#   },
#   "LSTM-ED": {
#     "Temp": "LSTM_Temp_1958_mae_seq2seq_v1",
#     "RH": "LSTM_RH_1958_mae_seq2seq_v1",
#     "WSpd": "LSTM_WSpd_1958_mae_seq2seq_v1",
#     "WDir": "LSTM_WDir_1958_mae_seq2seq_v1"
#   },
#   "LSTM-ED-FC": {
#     "Temp": "LSTM_Temp_1958_mae_seq2seq_v1_fc",
#     "RH": "LSTM_RH_1958_mae_seq2seq_v1_fc",
#     "WSpd": "LSTM_WSpd_1958_mae_seq2seq_v1_fc",
#     "WDir": "LSTM_WDir_1958_mae_seq2seq_v1_fc"
#   },
#   "LSTM-ED-ATTN-FC": {
#     "Temp": "LSTM_Temp_1958_mae_seq2seq_attn_fc",
#     "RH": "LSTM_RH_1958_mae_seq2seq_attn_fc",
#     "WSpd": "LSTM_WSpd_1958_mae_seq2seq_attn_fc",
#     "WDir": "LSTM_WDir_1958_mae_seq2seq_attn_fc"
#   },
#   "LSTM-ED-CNN-FC": {
#     "Temp": "LSTM_Temp_1958_mae_seq2seq_cnn_fc",
#     "RH": "LSTM_RH_1958_mae_seq2seq_cnn_fc",
#     "WSpd": "LSTM_WSpd_1958_mae_seq2seq_cnn_fc",
#     "WDir": "LSTM_WDir_1958_mae_seq2seq_cnn_fc"
#   },
#   "LSTM-ED-CNN-ATTN-FC": {
#     "Temp": "LSTM_Temp_1958_mae_seq2seq_cnn_attn_fc",
#     "RH": "LSTM_RH_1958_mae_seq2seq_cnn_attn_fc",
#     "WSpd": "LSTM_WSpd_1958_mae_seq2seq_cnn_attn_fc",
#     "WDir": "LSTM_WDir_1958_mae_seq2seq_cnn_attn_fc"
#   }
# }


# DAYS = {
#   "LSTM": {
#     "Temp": "LSTM_Temp_2662_mae_plain_lstm",
#     "RH": "LSTM_RH_2662_mae_plain_lstm",
#     "WSpd": "LSTM_WSpd_2662_mae_plain_lstm",
#     "WDir": "LSTM_WDir_2662_mae_plain_lstm"
#   },
#   "LSTM-FC": {
#     "Temp": "LSTM_Temp_2662_mae_plain_lstm_fc",
#     "RH": "LSTM_RH_2662_mae_plain_lstm_fc",
#     "WSpd": "LSTM_WSpd_2662_mae_plain_lstm_fc",
#     "WDir": "LSTM_WDir_2662_mae_plain_lstm_fc"
#   },
#   "LSTM-ED": {
#     "Temp": "LSTM_Temp_2662_mae_seq2seq_v1",
#     "RH": "LSTM_RH_2662_mae_seq2seq_v1",
#     "WSpd": "LSTM_WSpd_2662_mae_seq2seq_v1",
#     "WDir": "LSTM_WDir_2662_mae_seq2seq_v1"
#   },
#   "LSTM-ED-FC": {
#     "Temp": "LSTM_Temp_2662_mae_seq2seq_v1_fc",
#     "RH": "LSTM_RH_2662_mae_seq2seq_v1_fc",
#     "WSpd": "LSTM_WSpd_2662_mae_seq2seq_v1_fc",
#     "WDir": "LSTM_WDir_2662_mae_seq2seq_v1_fc"
#   },
#   "LSTM-ED-ATTN-FC": {
#     "Temp": "LSTM_Temp_2662_mae_seq2seq_attn_fc",
#     "RH": "LSTM_RH_2662_mae_seq2seq_attn_fc",
#     "WSpd": "LSTM_WSpd_2662_mae_seq2seq_attn_fc",
#     "WDir": "LSTM_WDir_2662_mae_seq2seq_attn_fc"
#   },
#   "LSTM-ED-CNN-FC": {
#     "Temp": "LSTM_Temp_2662_mae_seq2seq_cnn_fc",
#     "RH": "LSTM_RH_2662_mae_seq2seq_cnn_fc",
#     "WSpd": "LSTM_WSpd_2662_mae_seq2seq_cnn_fc",
#     "WDir": "LSTM_WDir_2662_mae_seq2seq_cnn_fc"
#   },
#   "LSTM-ED-CNN-ATTN-FC": {
#     "Temp": "LSTM_Temp_2662_mae_seq2seq_cnn_attn_fc",
#     "RH": "LSTM_RH_2662_mae_seq2seq_cnn_attn_fc",
#     "WSpd": "LSTM_WSpd_2662_mae_seq2seq_cnn_attn_fc",
#     "WDir": "LSTM_WDir_2662_mae_seq2seq_cnn_attn_fc"
#   }
# }

