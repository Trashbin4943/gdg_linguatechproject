from transformers import BertTokenizer, BertForSequenceClassification
import torch

'''
koBERT를 활용해 텍스트 감정을 분석하는 함수입니다.
감정 라벨은 0~12개로 일단은 늘려놨습니다.
ensemble.py에서 음성 감정 데이터랑 비교되는데 사용됩니다..
'''

def analyze_text_emotion(text):
    tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
    model = BertForSequenceClassification.from_pretrained('monologg/kobert', num_labels=12)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1)
    return int(pred.item())