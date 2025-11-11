from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

'''
1. STT를 거쳐 온 데이터를 RNN, LSTM, GRU 모델에 넣어 감정 분석
2. 각 모델 성능을 분석합니다.
3. 모델 성능을 비교하여 출력합니다.
'''

# RNN Model 
def build_rnn(input_dim, num_classes=12):
    model = Sequential()
    model.add(SimpleRNN(128, input_shape=(1, input_dim), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(SimpleRNN(64))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# LSTM model
def build_lstm(input_dim, num_classes=12):
    model = Sequential()
    model.add(LSTM(128, input_shape=(1, input_dim), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# GRU model
def build_gru(input_dim, num_classes=12):
    model = Sequential()
    model.add(GRU(128, input_shape=(1, input_dim), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(GRU(64))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 예측 
def predict_audio_emotion(model, features):
    features_seq = features.reshape(1, 1, -1)
    pred = model.predict(features_seq)
    return int(np.argmax(pred))

# 모델 평가
'''
정확도, F1-score, 혼동 행렬을 출력
'''
def evaluate_model(model, X_val_seq, y_val, model_name, label_map):
    y_pred = model.predict(X_val_seq)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_val, axis=1)

    # 정확도를 점수로 매겨 출력
    acc = accuracy_score(y_true_labels, y_pred_labels)
    print(f"\n🔍 {model_name} 정확도: {acc:.4f}")
    print(f"\n📋 {model_name} Classification Report:")
    print(classification_report(y_true_labels, y_pred_labels, target_names=label_map.keys()))

    cm = confusion_matrix(y_true_labels, y_pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_map.keys(),
                yticklabels=label_map.keys())
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()