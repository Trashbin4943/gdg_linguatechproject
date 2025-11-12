"""
KoBERT Fine-tuning for Complaint Classification
코랩에서 바로 실행 가능한 Fine-tuning 코드
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import os

# 라벨 매핑 정의
LABEL_MAPPING = {
    "정상": 0,
    "욕설_저주": 1,
    "모욕_조롱": 2,
    "폭력_위협_범죄조장": 3,
    "외설_성희롱": 4,
    "혐오표현": 5,
    "반복성": 6,
    "무리한_요구": 7,
    "부당성": 8,
    "허위_민원": 9,
    "장난전화": 10,
    "공포심_불안감_유발": 11
}

NUM_LABELS = len(LABEL_MAPPING)
REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}


class ComplaintDataset(Dataset):
    """민원 분류 데이터셋"""
    
    def __init__(self, texts: List[str], labels: List[List[int]], 
                 tokenizer: BertTokenizer, max_length: int = 128):
        """
        Args:
            texts: 텍스트 리스트
            labels: 다중 라벨 벡터 리스트 (이진 벡터)
            tokenizer: BERT 토크나이저
            max_length: 최대 시퀀스 길이
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }


def convert_labels_to_multilabel(label_strings: List[str]) -> List[List[int]]:
    """
    라벨 문자열 리스트를 다중 라벨 벡터로 변환
    
    Args:
        label_strings: ["욕설_저주|모욕_조롱", "정상", ...] 형식
    
    Returns:
        [[0, 1, 1, 0, ...], [1, 0, 0, 0, ...], ...] 형식
    """
    multilabel_vectors = []
    
    for label_str in label_strings:
        # "|"로 구분된 여러 라벨 처리
        labels = label_str.split('|') if isinstance(label_str, str) else [label_str]
        
        # 이진 벡터 생성
        vector = [0] * NUM_LABELS
        for label in labels:
            label = label.strip()
            if label in LABEL_MAPPING:
                vector[LABEL_MAPPING[label]] = 1
        
        multilabel_vectors.append(vector)
    
    return multilabel_vectors


def compute_metrics(eval_pred):
    """평가 메트릭 계산 (다중 라벨 분류)"""
    predictions, labels = eval_pred
    
    # 시그모이드 적용하여 확률로 변환
    predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
    
    # 임계값 0.5로 이진 분류
    binary_predictions = (predictions > 0.5).astype(int)
    
    # 각 라벨별 F1 점수 계산
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for i in range(labels.shape[1]):
        if labels[:, i].sum() > 0:  # 해당 라벨이 실제로 존재하는 경우만
            f1 = f1_score(labels[:, i], binary_predictions[:, i], 
                         average='binary', zero_division=0)
            precision = precision_recall_fscore_support(
                labels[:, i], binary_predictions[:, i], 
                average='binary', zero_division=0
            )[0]
            recall = precision_recall_fscore_support(
                labels[:, i], binary_predictions[:, i], 
                average='binary', zero_division=0
            )[1]
            
            f1_scores.append(f1)
            precision_scores.append(precision)
            recall_scores.append(recall)
    
    # 평균 점수
    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
    avg_precision = np.mean(precision_scores) if precision_scores else 0.0
    avg_recall = np.mean(recall_scores) if recall_scores else 0.0
    
    # 전체 정확도 (모든 라벨이 정확히 맞는 경우)
    exact_match = (binary_predictions == labels).all(axis=1).mean()
    
    return {
        'f1_macro': avg_f1,
        'precision_macro': avg_precision,
        'recall_macro': avg_recall,
        'exact_match': exact_match
    }


def split_by_session(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    """세션 단위로 데이터 분할 (세션 누수 방지)"""
    sessions = df['session_id'].unique()
    n_sessions = len(sessions)

    # 세션 단위로 분할
    train_sessions = sessions[:int(n_sessions * train_ratio)]
    val_sessions = sessions[int(n_sessions * train_ratio):
                           int(n_sessions * (train_ratio + val_ratio))]
    test_sessions = sessions[int(n_sessions * (train_ratio + val_ratio)):]

    train_df = df[df['session_id'].isin(train_sessions)]
    val_df = df[df['session_id'].isin(val_sessions)]
    test_df = df[df['session_id'].isin(test_sessions)]

    return train_df, val_df, test_df


def load_and_prepare_data(train_path: str, val_path: str = None, 
                         test_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    데이터 로드 및 전처리
    
    Args:
        train_path: 학습 데이터 경로 (CSV)
        val_path: 검증 데이터 경로 (CSV, 선택사항)
        test_path: 테스트 데이터 경로 (CSV, 선택사항)
    
    Returns:
        train_df, val_df, test_df
    """
    # 학습 데이터 로드
    train_df = pd.read_csv(train_path)
    
    # 라벨을 다중 라벨 벡터로 변환
    if 'label' in train_df.columns:
        train_df['multilabel_vector'] = convert_labels_to_multilabel(train_df['label'].tolist())
    
    # 검증 데이터 로드
    val_df = None
    if val_path and os.path.exists(val_path):
        val_df = pd.read_csv(val_path)
        if 'label' in val_df.columns:
            val_df['multilabel_vector'] = convert_labels_to_multilabel(val_df['label'].tolist())
    else:
        # 검증 데이터가 없으면 학습 데이터에서 분할
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            train_df, test_size=0.2, random_state=42, stratify=None
        )
    
    # 테스트 데이터 로드
    test_df = None
    if test_path and os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        if 'label' in test_df.columns:
            test_df['multilabel_vector'] = convert_labels_to_multilabel(test_df['label'].tolist())
    
    return train_df, val_df, test_df


def train_kobert_multilabel(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str = 'monologg/kobert',
    output_dir: str = './kobert_complaint_classifier',
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128
):
    """KoBERT 다중 라벨 분류 모델 학습"""
    print("=" * 80)
    print("KoBERT Fine-tuning 시작")
    print("=" * 80)
    
    # 토크나이저 로드
    print(f"\n1. 토크나이저 로드: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # 모델 로드
    print(f"2. 모델 로드: {model_name}")
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    )
    
    # 라벨 벡터화
    print("3. 라벨 벡터화")
    train_df['multilabel_vector'] = convert_labels_to_multilabel(train_df['label'].tolist())
    val_df['multilabel_vector'] = convert_labels_to_multilabel(val_df['label'].tolist())
    
    # 데이터셋 준비
    print("4. 데이터셋 준비")
    train_dataset = ComplaintDataset(
        train_df['text'].tolist(),
        train_df['multilabel_vector'].tolist(),
        tokenizer,
        max_length=max_length
    )
    
    val_dataset = ComplaintDataset(
        val_df['text'].tolist(),
        val_df['multilabel_vector'].tolist(),
        tokenizer,
        max_length=max_length
    )
    
    print(f"   - 학습 데이터: {len(train_dataset)}개")
    print(f"   - 검증 데이터: {len(val_dataset)}개")
    
    # 학습 설정
    print("5. 학습 설정")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        eval_strategy="epoch",  # Changed from evaluation_strategy
        save_strategy="epoch",  # "epoch": 매 에포크마다 저장, "steps": save_steps 간격으로 저장
        # save_steps=500,  # save_strategy="steps"일 때만 사용 (N 스텝마다 저장)
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=3,  # 최대 3개 체크포인트만 유지
    )
    
    # Trainer 초기화
    print("6. Trainer 초기화")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 학습 실행
    print("7. 학습 시작")
    print("-" * 80)
    trainer.train()
    print("-" * 80)
    
    # 모델 저장
    print(f"8. 모델 저장: {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # 최종 평가
    print("9. 최종 평가")
    eval_results = trainer.evaluate()
    print(f"   - F1 Macro: {eval_results['eval_f1_macro']:.4f}")
    print(f"   - Precision Macro: {eval_results['eval_precision_macro']:.4f}")
    print(f"   - Recall Macro: {eval_results['eval_recall_macro']:.4f}")
    print(f"   - Exact Match: {eval_results['eval_exact_match']:.4f}")
    
    print("\n" + "=" * 80)
    print("학습 완료!")
    print("=" * 80)
    
    return model, tokenizer, trainer


def predict_with_kobert(text: str, model, tokenizer, threshold: float = 0.5):
    """
    학습된 KoBERT 모델로 예측
    
    Args:
        text: 예측할 텍스트
        model: 학습된 모델
        tokenizer: 토크나이저
        threshold: 이진 분류 임계값
    
    Returns:
        예측된 라벨 리스트와 확률 딕셔너리
    """
    model.eval()
    
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=128
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze().numpy()
    
    # 임계값으로 이진 분류
    predictions = (probs > threshold).astype(int)
    
    # 라벨 매핑
    predicted_labels = [REVERSE_LABEL_MAPPING[i] for i in range(len(predictions)) 
                       if predictions[i] == 1]
    
    # 확률 딕셔너리
    prob_dict = {REVERSE_LABEL_MAPPING[i]: float(probs[i]) 
                for i in range(len(probs))}
    
    return predicted_labels, prob_dict


# 사용 예제
if __name__ == "__main__":
    # 예제: 데이터 로드 및 학습
    # train_df, val_df, test_df = load_and_prepare_data(
    #     train_path='train.csv',
    #     val_path='val.csv',
    #     test_path='test.csv'
    # )
    # 
    # # 또는 세션 단위로 분할
    # # train_df, val_df, test_df = split_by_session(train_df, train_ratio=0.7, val_ratio=0.15)
    # 
    # model, tokenizer, trainer = train_kobert_multilabel(
    #     train_df=train_df,
    #     val_df=val_df,
    #     epochs=3,
    #     batch_size=16
    # )
    # 
    # # 예측 예제
    # test_text = "X팔 너 거기서 뭐 배웠냐?"
    # labels, probs = predict_with_kobert(test_text, model, tokenizer)
    # print(f"예측 라벨: {labels}")
    # print(f"확률: {probs}")
    
    print("Fine-tuning 코드 준비 완료!")
    print("데이터셋을 준비한 후 위의 주석을 해제하여 실행하세요.")


