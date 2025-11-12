# Baseline 설계 및 Fine-tuning 가이드

## 📋 목차
1. [Baseline 설계 개요](#baseline-설계-개요)
2. [Baseline 아키텍처](#baseline-아키텍처)
3. [Fine-tuning을 위한 데이터셋 구조](#fine-tuning을-위한-데이터셋-구조)
4. [데이터 통합 과정](#데이터-통합-과정)
5. [KoBERT Fine-tuning 구현](#kobert-fine-tuning-구현)

---

## Baseline 설계 개요

### 🎯 설계 목표
- **빠른 프로토타입**: 제출 기한(11/28) 내 작동하는 결과물 확보
- **확장 가능성**: 이후 ML 모델 통합이 용이한 구조
- **실용성**: 실제 상담 데이터에 바로 적용 가능

### 📊 Baseline 구성 요소

#### 1. **키워드 기반 규칙 엔진** (Rule-based)
- **위치**: `classification_criteria.py`
- **방식**: 정규표현식 + 키워드 매칭
- **카테고리**: 11개 카테고리 + 정상 카테고리
  - 욕설_저주, 모욕_조롱, 폭력_위협_범죄조장, 외설_성희롱, 혐오표현
  - 반복성, 무리한_요구, 부당성, 허위_민원, 장난전화, 공포심_불안감_유발
- **장점**: 
  - 즉시 작동, 추가 학습 불필요
  - 명확한 판단 근거 제공 (설명 가능성)
  - 빠른 실행 속도
- **단점**:
  - 새로운 표현 패턴 감지 어려움
  - 완곡 표현/은어 처리 한계
  - 맥락 이해 부족

#### 2. **세션 맥락 분석** (Context-aware)
- **기능**: 이전 대화와 비교하여 반복성 감지
- **방식**: 간단한 키워드 유사도 + 반복 표현 패턴
- **향상 방안**: 문장 임베딩 유사도로 개선 예정

#### 3. **다중 카테고리 감지** (Multi-label)
- 한 텍스트에서 여러 문제 동시 감지
- 예: "X팔 너 거기서 뭐 배웠냐" → `욕설_저주` + `모욕_조롱`

#### 4. **심각도 기반 조치 제안** (Severity-based Action)
- 5단계 심각도 레벨
- 각 레벨별 자동 조치 방안 제시

---

## Baseline 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    입력 텍스트                            │
│              (STT 결과 또는 직접 입력)                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│          ClassificationCriteria.classify_text()         │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  키워드 매칭  │  │  세션 맥락   │  │  패턴 분석   │ │
│  │  (9개 카테고리)│  │  (반복성)    │  │  (심각도)    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  ClassificationResult 리스트 생성                 │  │
│  │  - category: ComplaintCategory                   │  │
│  │  - severity: ComplaintSeverity                   │  │
│  │  - confidence: float (0.0~1.0)                   │  │
│  │  - evidence: List[str] (판단 근거)                │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              다중 카테고리 결과                          │
│  [욕설_저주(HIGH), 모욕_조롱(MEDIUM), ...]              │
└─────────────────────────────────────────────────────────┘
```

### Baseline 키워드 확장

노트북 구현에서는 다음 키워드들이 추가로 포함되어 있습니다:

- **욕설 키워드 추가**: "닥쳐", "엿먹어", "개소리"
- **모욕 키워드 추가**: "쓰레기", "한심한", "어리석은", "수준 미달"
- **위협 키워드 추가**: "가만 안 두겠다", "해코지", "신상 털어", "가만두지 않겠다"
- **반복성 키워드 추가**: "같은 내용", "몇 번을 말해야", "똑같은 이야기"

### 현재 Baseline의 한계점

1. **표현 다양성 부족**
   - 키워드 리스트가 제한적 (지속적 확장 필요)
   - 완곡 표현, 은어, 신조어 미처리

2. **맥락 이해 부족**
   - 문맥에 따른 의미 변화 미반영
   - 예: "죽여줘" (농담 vs 진지한 위협)

3. **반복성 감지 정확도 낮음**
   - 단순 키워드 매칭으로는 한계
   - 의미적 유사도 측정 필요

4. **무리한 요구/부당성 판단 어려움**
   - 도메인 지식(매뉴얼, 권한) 필요
   - 규칙만으로는 판단 한계

---

## Fine-tuning을 위한 데이터셋 구조

### 📁 데이터셋 형식

#### 옵션 1: CSV 형식 (권장)
```csv
text,label,severity,session_id,turn_id,context
"X팔 너 거기서 뭐 배웠냐?",욕설_저주|모욕_조롱,HIGH|MEDIUM,session_001,1,
"앞선 통화에서도 말씀드렸다시피 같은 얘기인데요",반복성,MEDIUM,session_001,2,"이전 대화 내용"
"정상적인 문의입니다",정상,NORMAL,session_002,1,
```

#### 옵션 2: JSON 형식
```json
{
  "sessions": [
    {
      "session_id": "session_001",
      "turns": [
        {
          "turn_id": 1,
          "speaker": "customer",
          "text": "X팔 너 거기서 뭐 배웠냐?",
          "labels": ["욕설_저주", "모욕_조롱"],
          "severities": ["HIGH", "MEDIUM"],
          "context": []
        },
        {
          "turn_id": 2,
          "speaker": "customer",
          "text": "앞선 통화에서도 말씀드렸다시피",
          "labels": ["반복성"],
          "severities": ["MEDIUM"],
          "context": ["이전 대화"]
        }
      ]
    }
  ]
}
```

### 🏷️ 라벨링 체계

#### 다중 라벨 (Multi-label) 구조
- **하나의 텍스트에 여러 카테고리 동시 라벨링 가능**
- 예: `["욕설_저주", "모욕_조롱"]`

#### 라벨 매핑 (KoBERT 입력용)
```python
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

# 다중 라벨을 위한 이진 벡터
# 예: ["욕설_저주", "모욕_조롱"] → [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

### 📊 데이터셋 요구사항

#### 최소 데이터 규모
- **카테고리별 최소 100개 이상** (권장: 500개 이상)
- **총 데이터: 최소 1,000개** (권장: 5,000개 이상)
- **클래스 불균형 고려**: 소수 클래스도 최소 50개 이상

#### 데이터 분할
```
전체 데이터 (100%)
├── 학습 데이터 (train): 70%
├── 검증 데이터 (val): 15%
└── 테스트 데이터 (test): 15%
```

#### 세션 누수 방지
- **같은 세션의 데이터는 같은 split에만 포함**
- 세션 단위로 분할 필요

---

## 데이터 통합 과정

### 1단계: 데이터 수집 및 정제

```python
# data_preparation.py

import pandas as pd
import json
from typing import List, Dict

def load_raw_data(data_path: str, format: str = "csv"):
    """원본 데이터 로드"""
    if format == "csv":
        df = pd.read_csv(data_path)
    elif format == "json":
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = convert_json_to_dataframe(data)
    return df

def clean_text(text: str) -> str:
    """텍스트 정제"""
    # 개인정보 마스킹
    text = mask_pii(text)
    # 특수문자 정규화
    text = normalize_special_chars(text)
    # 공백 정리
    text = ' '.join(text.split())
    return text

def convert_labels_to_multilabel(labels: str) -> List[int]:
    """라벨 문자열을 다중 라벨 벡터로 변환"""
    label_list = labels.split('|')
    multilabel_vector = [0] * len(LABEL_MAPPING)
    for label in label_list:
        if label in LABEL_MAPPING:
            multilabel_vector[LABEL_MAPPING[label]] = 1
    return multilabel_vector
```

### 2단계: Baseline으로 자동 라벨링 (부족한 데이터 보완)

```python
# auto_labeling.py

from classification_criteria import ClassificationCriteria

def auto_label_with_baseline(text: str, session_context: List[str] = None):
    """Baseline 규칙으로 자동 라벨링 (검증용)"""
    results = ClassificationCriteria.classify_text(text, session_context)
    
    # 정상이 아닌 결과만 추출
    labels = [r.category.value for r in results if r.severity.value != "정상"]
    
    if not labels:
        return ["정상"]
    return labels

def augment_dataset_with_baseline(df: pd.DataFrame):
    """Baseline으로 라벨이 없는 데이터에 자동 라벨링"""
    df['auto_labels'] = df.apply(
        lambda row: auto_label_with_baseline(row['text'], row.get('context', [])),
        axis=1
    )
    return df
```

### 3단계: 데이터 검증 및 품질 관리

```python
# data_validation.py

def validate_dataset(df: pd.DataFrame):
    """데이터셋 품질 검증"""
    issues = []
    
    # 1. 빈 텍스트 체크
    empty_texts = df[df['text'].str.strip() == '']
    if len(empty_texts) > 0:
        issues.append(f"빈 텍스트 {len(empty_texts)}건 발견")
    
    # 2. 라벨 분포 체크
    label_counts = count_labels(df)
    for label, count in label_counts.items():
        if count < 50:
            issues.append(f"라벨 '{label}' 샘플 부족: {count}개")
    
    # 3. 세션 누수 체크
    session_leakage = check_session_leakage(df)
    if session_leakage:
        issues.append("세션 누수 발견")
    
    return issues

def split_by_session(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    """세션 단위로 데이터 분할"""
    sessions = df['session_id'].unique()
    n_sessions = len(sessions)
    
    train_sessions = sessions[:int(n_sessions * train_ratio)]
    val_sessions = sessions[int(n_sessions * train_ratio):
                           int(n_sessions * (train_ratio + val_ratio))]
    test_sessions = sessions[int(n_sessions * (train_ratio + val_ratio)):]
    
    train_df = df[df['session_id'].isin(train_sessions)]
    val_df = df[df['session_id'].isin(val_sessions)]
    test_df = df[df['session_id'].isin(test_sessions)]
    
    return train_df, val_df, test_df
```

### 4단계: KoBERT 입력 형식 변환

```python
# data_conversion.py

from transformers import BertTokenizer

def prepare_kobert_dataset(df: pd.DataFrame, tokenizer: BertTokenizer, max_length=128):
    """KoBERT 학습용 데이터셋 준비"""
    
    texts = df['text'].tolist()
    labels = df['multilabel_vector'].tolist()  # 이진 벡터 리스트
    
    # 토크나이징
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # 라벨을 텐서로 변환
    import torch
    label_tensors = torch.tensor(labels, dtype=torch.float)
    
    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': label_tensors
    }
```

---

## KoBERT Fine-tuning 구현

### 아키텍처 설계

```
┌─────────────────────────────────────────────────────────┐
│              Baseline (Rule-based)                      │
│         빠른 필터링, 명확한 근거 제공                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         KoBERT Fine-tuned Model                         │
│    맥락 이해, 다양한 표현 패턴 감지                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Ensemble (하이브리드)                       │
│  Baseline + KoBERT 결과 통합                            │
│  - Baseline 신뢰도 높으면 Baseline 우선                  │
│  - 모호한 경우 KoBERT 결과 활용                          │
└─────────────────────────────────────────────────────────┘
```

### Fine-tuning 코드 구조

```python
# kobert_finetuning.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import f1_score, precision_recall_fscore_support
import numpy as np

class ComplaintDataset(Dataset):
    """민원 분류 데이터셋"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
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

def compute_metrics(eval_pred):
    """평가 메트릭 계산 (다중 라벨)"""
    predictions, labels = eval_pred
    
    # 시그모이드 적용하여 이진 분류로 변환
    predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
    predictions = (predictions > 0.5).astype(int)
    
    # 각 라벨별 F1 점수
    f1_scores = []
    for i in range(labels.shape[1]):
        f1 = f1_score(labels[:, i], predictions[:, i], average='binary', zero_division=0)
        f1_scores.append(f1)
    
    # 평균 F1 점수
    avg_f1 = np.mean(f1_scores)
    
    return {
        'f1_macro': avg_f1,
        'f1_per_label': {f'label_{i}': f1 for i, f1 in enumerate(f1_scores)}
    }

def train_kobert_multilabel(
    train_df,
    val_df,
    model_name='monologg/kobert',
    num_labels=12,
    output_dir='./kobert_complaint_classifier',
    epochs=3,
    batch_size=16,
    learning_rate=2e-5
):
    """KoBERT 다중 라벨 분류 모델 학습"""
    
    # 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # 모델 로드 (다중 라벨 분류용)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"  # 다중 라벨 설정
    )
    
    # 데이터셋 준비
    train_dataset = ComplaintDataset(
        train_df['text'].tolist(),
        train_df['multilabel_vector'].tolist(),
        tokenizer
    )
    
    val_dataset = ComplaintDataset(
        val_df['text'].tolist(),
        val_df['multilabel_vector'].tolist(),
        tokenizer
    )
    
    # 학습 설정
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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
    )
    
    # Trainer 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 학습 실행
    trainer.train()
    
    # 모델 저장
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer
```

### Risk Score 계산 방법론

Risk Score는 여러 악성 요인을 종합하여 0-10점 척도로 위험도를 산정합니다. 다양한 계산 방법론을 제시하며, 실제 구현에서는 데이터 특성과 요구사항에 맞게 선택할 수 있습니다.

#### 기본 심각도 점수 매핑

모든 방법론에서 공통으로 사용하는 심각도별 기본 점수:

- **CRITICAL**: 4점
- **HIGH**: 3점
- **MEDIUM**: 2점
- **LOW**: 1점
- **NORMAL**: 0점

#### 방법론 1: 선형 합산 방식 (Linear Sum)

가장 단순하고 직관적인 방식으로, 모든 악성 요인의 점수를 단순 합산합니다.

```python
def calculate_risk_linear(baseline_issues, metadata_issues):
    """선형 합산 방식"""
    total_score = 0
    
    # Baseline 이슈 점수 합산
    for issue in baseline_issues:
        if 'CRITICAL' in issue: total_score += 4
        elif 'HIGH' in issue: total_score += 3
        elif 'MEDIUM' in issue: total_score += 2
        elif 'LOW' in issue: total_score += 1
    
    # 메타데이터 이슈 점수 합산
    for issue in metadata_issues:
        if '고충 상담' in issue: total_score += 3
        elif '해결 불가' in issue: total_score += 3
        # ... 기타 메타데이터 점수
    
    return min(total_score, 10)
```

**장점**: 
- 구현이 간단하고 이해하기 쉬움
- 점수 계산이 투명함
- 예측 가능한 결과

**단점**: 
- 여러 요인이 중첩되어도 선형적으로만 증가
- 심각한 케이스와 경미한 케이스의 구분이 약함

---

#### 방법론 2: 지수적 증폭 방식 (Exponential Amplification)

합산된 점수를 거듭제곱하여 증폭하는 방식으로, 여러 악성 요인이 동시에 감지될 때 위험도를 더 크게 반영합니다.

```python
def calculate_risk_exponential(baseline_issues, metadata_issues, power=1.5):
    """지수적 증폭 방식"""
    linear_sum = calculate_risk_linear(baseline_issues, metadata_issues)
    
    # 지수적 증폭
    amplified_score = linear_sum ** power
    
    # 0-10 스케일 조정
    final_score = min(round(amplified_score), 10)
    
    # 최소 1점 보장 (악성 요인 감지 시)
    if linear_sum > 0 and final_score == 0:
        final_score = 1
    
    return final_score
```

**장점**: 
- 여러 악성 요인 중첩 시 위험도가 크게 증가
- 심각한 케이스를 더 명확히 구분
- `power` 파라미터로 증폭 강도 조절 가능

**단점**: 
- 점수 변화가 비선형적이라 예측이 어려움
- `power` 값에 따라 결과가 크게 달라짐
- 경미한 케이스도 과도하게 증폭될 수 있음

---

#### 방법론 3: 가중 평균 방식 (Weighted Average)

Baseline 점수와 메타데이터 점수에 가중치를 적용하여 평균을 계산하는 방식입니다.

```python
def calculate_risk_weighted(baseline_score, metadata_score, 
                           baseline_weight=0.7, metadata_weight=0.3):
    """가중 평균 방식"""
    weighted_score = (baseline_score * baseline_weight + 
                     metadata_score * metadata_weight)
    
    return min(round(weighted_score), 10)
```

**장점**: 
- Baseline과 메타데이터의 중요도를 조절 가능
- 두 점수 소스의 균형을 맞출 수 있음
- 도메인 특성에 맞게 가중치 조정 가능

**단점**: 
- 가중치 설정이 주관적일 수 있음
- 최대값보다 낮은 점수가 나올 수 있음

---

#### 방법론 4: 최대값 방식 (Maximum)

Baseline 점수와 메타데이터 점수 중 더 높은 값을 선택하는 방식입니다.

```python
def calculate_risk_maximum(baseline_score, metadata_score):
    """최대값 방식"""
    return max(baseline_score, metadata_score)
```

**장점**: 
- 가장 심각한 요인을 우선 반영
- 구현이 매우 간단
- 보수적인 위험도 평가

**단점**: 
- 여러 요인의 중첩 효과를 반영하지 못함
- 점수가 낮게 나올 수 있음

---

#### 방법론 5: 로그 스케일 방식 (Logarithmic Scale)

선형 합산 점수에 로그 함수를 적용하여 점수 증가를 완화하는 방식입니다.

```python
import math

def calculate_risk_logarithmic(baseline_issues, metadata_issues, base=2):
    """로그 스케일 방식"""
    linear_sum = calculate_risk_linear(baseline_issues, metadata_issues)
    
    if linear_sum == 0:
        return 0
    
    # 로그 스케일 적용
    log_score = math.log(linear_sum + 1, base) * (10 / math.log(11, base))
    
    return min(round(log_score), 10)
```

**장점**: 
- 점수 증가가 완만함
- 낮은 점수 구간에서 세밀한 구분 가능
- 높은 점수 구간에서 포화 효과

**단점**: 
- 높은 위험도 케이스의 구분이 약함
- 로그 함수의 특성상 직관적이지 않음

---

#### 방법론 6: 단계별 점수 방식 (Tiered Scoring)

심각도 레벨에 따라 단계별로 점수를 부여하는 방식입니다.

```python
def calculate_risk_tiered(baseline_issues, metadata_issues):
    """단계별 점수 방식"""
    # 가장 높은 심각도 레벨 찾기
    max_severity = 0
    
    for issue in baseline_issues:
        if 'CRITICAL' in issue:
            max_severity = max(max_severity, 4)
        elif 'HIGH' in issue:
            max_severity = max(max_severity, 3)
        elif 'MEDIUM' in issue:
            max_severity = max(max_severity, 2)
        elif 'LOW' in issue:
            max_severity = max(max_severity, 1)
    
    # 이슈 개수에 따른 보너스 점수
    issue_count = len(baseline_issues) + len(metadata_issues)
    bonus = min(issue_count - 1, 2)  # 최대 2점 보너스
    
    return min(max_severity + bonus, 10)
```

**장점**: 
- 가장 심각한 요인을 우선 반영
- 여러 요인 중첩 시 보너스 점수 부여
- 직관적이고 해석하기 쉬움

**단점**: 
- 보너스 점수 계산 방식이 주관적
- 세밀한 점수 구분이 어려움

---

#### 방법론 7: 하이브리드 방식 (Hybrid)

여러 방법론을 조합하여 사용하는 방식입니다.

```python
def calculate_risk_hybrid(baseline_issues, metadata_issues, method='adaptive'):
    """하이브리드 방식"""
    baseline_score = calculate_risk_linear(baseline_issues, [])
    metadata_score = calculate_risk_linear([], metadata_issues)
    
    if method == 'adaptive':
        # 점수가 낮으면 선형, 높으면 지수적 증폭
        if baseline_score + metadata_score < 5:
            return calculate_risk_linear(baseline_issues, metadata_issues)
        else:
            return calculate_risk_exponential(baseline_issues, metadata_issues)
    
    elif method == 'weighted_max':
        # 가중 평균과 최대값의 평균
        weighted = calculate_risk_weighted(baseline_score, metadata_score)
        maximum = calculate_risk_maximum(baseline_score, metadata_score)
        return round((weighted + maximum) / 2)
```

**장점**: 
- 여러 방법론의 장점을 결합
- 상황에 따라 적응적으로 계산
- 유연한 점수 산정

**단점**: 
- 구현이 복잡함
- 결과 해석이 어려울 수 있음

---

#### 방법론 선택 가이드

| 방법론 | 적합한 상황 | 권장 파라미터 |
|--------|-----------|--------------|
| 선형 합산 | 단순하고 투명한 점수 산정 필요 | - |
| 지수적 증폭 | 여러 요인 중첩 시 위험도 강조 | power=1.5~2.0 |
| 가중 평균 | Baseline과 메타데이터 균형 필요 | baseline_weight=0.6~0.8 |
| 최대값 | 가장 심각한 요인만 반영 | - |
| 로그 스케일 | 낮은 점수 구간 세밀한 구분 필요 | base=2~3 |
| 단계별 점수 | 심각도 레벨 중심 평가 | bonus_max=2~3 |
| 하이브리드 | 복잡한 요구사항, 적응적 평가 | method='adaptive' |

#### 실제 구현 권장사항

1. **초기 구현**: 선형 합산 방식으로 시작하여 점수 분포 확인
2. **데이터 분석**: 실제 데이터에서 점수 분포와 레벨 분리도 확인
3. **방법론 선택**: 데이터 특성과 비즈니스 요구사항에 맞는 방법론 선택
4. **파라미터 튜닝**: 선택한 방법론의 파라미터를 검증 데이터로 튜닝
5. **지속적 개선**: 운영 데이터를 바탕으로 방법론 개선

**참고**: 현재 노트북 구현에서는 지수적 증폭 방식을 사용하고 있으며, `amplification_power=1.5`로 설정되어 있습니다. 필요에 따라 다른 방법론으로 변경하거나 하이브리드 방식으로 확장할 수 있습니다.

### 하이브리드 통합 (Baseline + KoBERT)

```python
# hybrid_classifier.py

from classification_criteria import ClassificationCriteria
from risk_based_classifier import RiskScoreClassifier
from transformers import BertTokenizer, BertForSequenceClassification
import torch

class HybridComplaintClassifier:
    """Baseline + KoBERT 하이브리드 분류기"""
    
    def __init__(self, kobert_model_path: str = None, kobert_model=None, tokenizer=None):
        # Baseline 규칙 엔진
        self.baseline = ClassificationCriteria()
        self.risk_classifier = RiskScoreClassifier()
        
        # KoBERT 모델 (선택사항)
        self.kobert_model = kobert_model
        self.tokenizer = tokenizer
        self.use_kobert = kobert_model is not None and tokenizer is not None
        
        if kobert_model_path and not self.use_kobert:
            try:
                self.tokenizer = BertTokenizer.from_pretrained(kobert_model_path)
                self.kobert_model = BertForSequenceClassification.from_pretrained(kobert_model_path)
                self.kobert_model.eval()
                self.use_kobert = True
            except Exception as e:
                print(f"⚠️ KoBERT 모델 로드 실패: {e}")
                self.use_kobert = False
    
    def classify(self, text: str, session_context: List[str] = None,
                 metadata: Optional[ConsultationMetadata] = None,
                 use_baseline_threshold: float = 0.8):
        """
        하이브리드 분류
        
        Args:
            text: 분석할 텍스트
            session_context: 세션 맥락
            metadata: 상담 메타데이터
            use_baseline_threshold: Baseline 신뢰도가 이 값 이상이면 Baseline 우선 사용
        """
        # 1. Baseline 분류
        baseline_results = self.baseline.classify_text(text, session_context)
        risk_result = self.risk_classifier.classify(text, session_context, metadata)
        baseline_max_confidence = max([r.confidence for r in baseline_results], default=0.0)
        
        # 2. Baseline 신뢰도가 높으면 Baseline 결과 사용
        if baseline_max_confidence >= use_baseline_threshold:
            return {
                'method': 'baseline',
                'risk_score': risk_result.risk_score,
                'risk_level': risk_result.risk_level.name,
                'labels': [r.category.value for r in baseline_results if r.severity != ComplaintSeverity.NORMAL],
                'confidence': baseline_max_confidence,
                'recommendation': risk_result.recommendation
            }
        
        # 3. 그렇지 않으면 KoBERT 사용
        if self.use_kobert:
            kobert_labels, kobert_probs = self._classify_with_kobert(text)
            
            # 4. 두 결과 통합 (Ensemble)
            ensemble_labels = set()
            
            # KoBERT 결과 추가
            for label in kobert_labels:
                ensemble_labels.add(label)
            
            # Baseline의 높은 신뢰도 결과 추가
            for result in baseline_results:
                if result.confidence > 0.7:
                    ensemble_labels.add(result.category.value)
            
            return {
                'method': 'hybrid',
                'risk_score': risk_result.risk_score,
                'risk_level': risk_result.risk_level.name,
                'labels': list(ensemble_labels),
                'baseline_labels': [r.category.value for r in baseline_results if r.severity != ComplaintSeverity.NORMAL],
                'kobert_labels': kobert_labels,
                'kobert_probs': kobert_probs,
                'confidence': max(baseline_max_confidence, max(kobert_probs.values()) if kobert_probs else 0.0),
                'recommendation': risk_result.recommendation
            }
        else:
            # KoBERT가 없으면 Baseline만 사용
            return {
                'method': 'baseline',
                'risk_score': risk_result.risk_score,
                'risk_level': risk_result.risk_level.name,
                'labels': [r.category.value for r in baseline_results if r.severity != ComplaintSeverity.NORMAL],
                'confidence': baseline_max_confidence,
                'recommendation': risk_result.recommendation
            }
    
    def _classify_with_kobert(self, text: str, threshold: float = 0.5):
        """KoBERT로 분류"""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        )
        
        with torch.no_grad():
            outputs = self.kobert_model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits).squeeze().numpy()
        
        # 임계값으로 이진 분류
        predictions = (probs > threshold).astype(int)
        
        # 라벨 매핑 역변환
        predicted_labels = [REVERSE_LABEL_MAPPING[i] for i in range(len(predictions))
                           if predictions[i] == 1]
        
        # 확률 딕셔너리
        prob_dict = {REVERSE_LABEL_MAPPING[i]: float(probs[i])
                    for i in range(len(probs))}
        
        return predicted_labels, prob_dict
```

---

## 📝 데이터셋 준비 체크리스트

### 필수 항목
- [ ] 원본 데이터 수집 (CSV/JSON)
- [ ] 텍스트 정제 (PII 마스킹, 특수문자 정규화)
- [ ] 라벨링 (다중 라벨 지원)
- [ ] 세션 ID 할당
- [ ] 데이터 분할 (세션 누수 방지)
- [ ] 클래스 불균형 확인 및 처리

### 권장 항목
- [ ] Baseline으로 자동 라벨링 (검증용)
- [ ] 데이터 품질 검증
- [ ] 통계 분석 (라벨 분포, 텍스트 길이 등)
- [ ] 샘플 데이터 시각화

---

## 🚀 다음 단계

1. **데이터셋 확인**: 제공된 데이터셋 구조 파악
2. **데이터 전처리**: 위의 통합 과정 적용
3. **Baseline 검증**: 현재 규칙 엔진으로 샘플 테스트
4. **KoBERT Fine-tuning**: 데이터 준비 완료 후 학습
5. **하이브리드 통합**: Baseline + KoBERT 결합
6. **평가 및 개선**: 성능 측정 및 반복 개선


