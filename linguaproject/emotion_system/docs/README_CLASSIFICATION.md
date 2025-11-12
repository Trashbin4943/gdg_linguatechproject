# 악성 민원 분류 시스템 - Baseline & Fine-tuning 가이드

## 🎯 빠른 시작

### Baseline 사용 (즉시 작동)
```python
from classification_criteria import ClassificationCriteria

text = "X팔 너 거기서 뭐 배웠냐?"
results = ClassificationCriteria.classify_text(text)

for result in results:
    print(f"{result.category.value}: {result.severity.value} (신뢰도: {result.confidence:.2f})")
```

### Fine-tuning 준비
1. 데이터셋 준비 (CSV 형식)
2. `kobert_finetuning.py` 실행
3. 학습된 모델로 예측

---

## 📊 Baseline 설계 요약

### 구조
- **키워드 기반 규칙 엔진**: 11개 카테고리 + 정상 카테고리, 즉시 작동
  - 욕설_저주, 모욕_조롱, 폭력_위협_범죄조장, 외설_성희롱, 혐오표현
  - 반복성, 무리한_요구, 부당성, 허위_민원, 장난전화, 공포심_불안감_유발
- **세션 맥락 분석**: 반복성 감지 (키워드 유사도 + 반복 표현 패턴)
- **다중 라벨 감지**: 한 텍스트에서 여러 문제 동시 감지
- **심각도 기반 조치**: 5단계 레벨 (NORMAL, LOW, MEDIUM, HIGH, CRITICAL), 자동 조치 제안
- **Risk Score 계산**: 여러 방법론 중 선택 가능 (선형 합산, 지수적 증폭, 가중 평균, 최대값, 로그 스케일, 단계별 점수, 하이브리드 등)
  - 현재 노트북 구현: 지수적 증폭 방식 (`amplification_power=1.5`)
  - 상세 방법론: `BASELINE_AND_FINETUNING.md` 참조

### 장점
✅ 즉시 작동 (학습 불필요)  
✅ 명확한 판단 근거 제공  
✅ 빠른 실행 속도  
✅ 설명 가능성 (XAI)

### 한계
❌ 새로운 표현 패턴 감지 어려움  
❌ 완곡 표현/은어 처리 한계  
❌ 맥락 이해 부족

---

## 🔄 Fine-tuning을 위한 데이터셋

### 📋 데이터 형식 선호도
1. **CSV 형식** (가장 선호) - pandas로 바로 로드 가능
2. JSON 형식 - 구조화된 데이터 표현
3. Excel 형식 - CSV로 변환 필요

### ✅ 필수 사항

#### 1. 필수 컬럼
- `text`: 텍스트 내용 (UTF-8, NULL 없음, 최소 1자)
- `label`: 라벨 (유효한 라벨 값, NULL 없음)

#### 2. 라벨 형식
- **단일 라벨**: `"욕설_저주"`
- **다중 라벨**: `"욕설_저주|모욕_조롱"` (파이프 `|`로 구분)

#### 3. 허용되는 라벨 값
```
정상, 욕설_저주, 모욕_조롱, 폭력_위협_범죄조장, 외설_성희롱,
혐오표현, 반복성, 무리한_요구, 부당성, 허위_민원, 장난전화,
공포심_불안감_유발
```

#### 4. 최소 데이터 규모
- **전체**: 최소 500개 (권장: 5,000개)
- **카테고리별**: 최소 20개 (권장: 100개)
- **정상 샘플**: 전체의 30% 이상

### ⭐ 권장 사항

#### 권장 컬럼
- `session_id`: 세션 식별자 (반복성 감지용)
- `turn_id`: 턴 번호 (대화 순서)
- `severity`: 심각도 (NORMAL, LOW, MEDIUM, HIGH, CRITICAL)
- `speaker`: 화자 (customer, agent)
- `context`: 맥락 정보 (이전 대화)

### 📝 데이터 예시

#### 최소 형식
```csv
text,label
"정상적인 문의입니다",정상
"X팔 너 거기서 뭐 배웠냐?",욕설_저주|모욕_조롱
"앞선 통화에서도 말씀드렸다시피",반복성
```

#### 권장 형식
```csv
text,label,session_id,turn_id,severity,speaker
"정상적인 문의입니다",정상,session_001,1,NORMAL,customer
"X팔 너 거기서 뭐 배웠냐?",욕설_저주|모욕_조롱,session_001,2,HIGH|MEDIUM,customer
"앞선 통화에서도 말씀드렸다시피",반복성,session_002,1,MEDIUM,customer
```

### 🔍 데이터 검증
```python
from dataset_validator import print_validation_report

# 데이터셋 검증
print_validation_report('your_dataset.csv')
```

**상세 가이드**: `DATASET_REQUIREMENTS.md` 참조

---

## 🚀 통합 과정

### 1단계: 데이터 수집 및 정제
- 원본 데이터 로드 (CSV/JSON)
- 텍스트 정제 (PII 마스킹, 특수문자 정규화)
- 라벨 변환 (다중 라벨 벡터)

### 2단계: Baseline으로 자동 라벨링 (선택)
- 라벨이 없는 데이터에 Baseline 적용
- 검증용 데이터 생성

### 3단계: 데이터 검증
- 빈 텍스트 체크
- 라벨 분포 확인
- 세션 누수 방지 (세션 단위 분할)

### 4단계: KoBERT Fine-tuning
- 다중 라벨 분류 모델 학습
- F1, Precision, Recall 평가

### 5단계: 하이브리드 통합
- Baseline + KoBERT Ensemble
- Risk Score 계산 (다양한 방법론 중 선택 가능)
- Baseline 신뢰도 높으면 Baseline 우선 (기본 임계값: 0.8)
- 모호한 경우 KoBERT 결과 활용
- 최종 결과에 Risk Score 및 권장 조치 포함

**Risk Score 계산 방법론**: 선형 합산, 지수적 증폭, 가중 평균, 최대값, 로그 스케일, 단계별 점수, 하이브리드 등 다양한 방법론을 제시하며, 데이터 특성과 요구사항에 맞게 선택할 수 있습니다. 상세 내용은 `BASELINE_AND_FINETUNING.md`의 "Risk Score 계산 방법론" 섹션을 참조하세요.

---

## 📁 파일 구조

```
emotion_system/
├── classification_criteria.py      # Baseline 규칙 엔진
├── test_classification.py          # Baseline 테스트
├── classification_usage_example.py # 사용 예제
├── kobert_finetuning.py           # KoBERT Fine-tuning 코드
├── BASELINE_AND_FINETUNING.md     # 상세 가이드
└── README_CLASSIFICATION.md       # 이 파일
```

---

## 🔧 다음 단계

1. **데이터셋 확인**: 제공된 데이터셋 구조 파악
2. **Baseline 검증**: 샘플 데이터로 테스트
3. **데이터 전처리**: CSV 형식으로 변환
4. **KoBERT Fine-tuning**: 학습 실행
5. **하이브리드 통합**: Baseline + KoBERT 결합
6. **평가 및 개선**: 성능 측정 및 반복 개선

---

## 📚 상세 문서

- **Baseline 설계**: `BASELINE_AND_FINETUNING.md` 참조
- **코드 예제**: `classification_usage_example.py` 참조
- **Fine-tuning 코드**: `kobert_finetuning.py` 참조

