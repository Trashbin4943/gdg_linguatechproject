from audio_features import extract_features
from audio_models import build_rnn, build_lstm, build_gru, evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
import os

'''
audio_folder(임시 학습 데이터 폴더)의 .wav 파일을 기반으로 음향 특징을 추출하고 
audio_models.py에서 정의한 RNN, LSTM, GRU 모델을 학습합니다.
RNN, LSTM, GRU 모델을 비교 평가하는데 사용됩니다.
'''

label_map = {
'불만': 0, '분노': 1, '불안': 2, '중립': 3,
'감사': 4, '요청': 5, '혼란': 6
}

# 데이터 셋 생성
def build_dataset(audio_dir, label_map):
    data, labels = []
    for filename in os.listdir(audio_dir):
        if filename.endswith('.wav'):
            path = os.path.join(audio_dir, filename)
            features = extract_features(path)
            data.append(features)
            emotion = filename.split('_')[0]
            labels.append(label_map[emotion])
    df = pd.DataFrame(data)
    return df, labels

# 데이터 폴더 이름 바꾸시면 여기⬇ 수정해주세요
X_df, y = build_dataset('audio_folder', label_map)
scaler = StandardScaler()
X = scaler.fit_transform(X_df)
y_cat = to_categorical(y)
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)
X_train_seq = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_val_seq = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])

rnn = build_rnn(X_train.shape[1], len(label_map))
lstm = build_lstm(X_train.shape[1], len(label_map))
gru = build_gru(X_train.shape[1], len(label_map))

rnn.fit(X_train_seq, y_train, epochs=30, batch_size=16, validation_data=(X_val_seq, y_val))
lstm.fit(X_train_seq, y_train, epochs=30, batch_size=16, validation_data=(X_val_seq, y_val))
gru.fit(X_train_seq, y_train, epochs=30, batch_size=16, validation_data=(X_val_seq, y_val))

# 평가
evaluate_model(rnn, X_val_seq, y_val, "RNN", label_map)
evaluate_model(lstm, X_val_seq, y_val, "LSTM", label_map)
evaluate_model(gru, X_val_seq, y_val, "GRU", label_map)