'''
LSTM 기반 음향 감정 분석
음향 특징 벡터 (MFCC, pitch, energy 등)를 입력받아 
감정라벨을 출력합니다.
'''

import torch
import torch.nn as nn
from .label_map import label_map

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn.squeeze(0))

def classify_audio_emotion(features):
    model = SimpleLSTM(input_dim=features.shape[2], num_classes=len(label_map))
    model.eval()
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        logits = model(x)
        label = torch.argmax(logits, dim=1).item()
        return label_map[label]