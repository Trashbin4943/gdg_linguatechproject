'''
KoBERT 기반 텍스트 감정 분석
발화 텍스트를 입력받아 감정 라벨을 출력합니다.
'''

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from .label_map import label_map

def classify_text_emotion(text):
    tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
    model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=len(label_map))
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    label = torch.argmax(outputs.logits, dim=1).item()
    return label_map[label]