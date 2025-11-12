"""
데이터셋 검증 및 로드 유틸리티

데이터셋의 형식과 품질을 검증하고, 필요한 형식으로 변환하는 도구
"""

import pandas as pd
import json
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path

# 라벨 매핑 (classification_criteria.py와 동일)
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

VALID_LABELS = set(LABEL_MAPPING.keys())
VALID_SEVERITIES = {"NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"}


class DatasetValidator:
    """데이터셋 검증 클래스"""
    
    def __init__(self, required_columns: List[str] = None):
        """
        Args:
            required_columns: 필수 컬럼 리스트 (기본: ['text', 'label'])
        """
        self.required_columns = required_columns or ['text', 'label']
        self.errors = []
        self.warnings = []
    
    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
        """
        데이터셋 검증
        
        Returns:
            (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # 1. 필수 컬럼 확인
        self._check_required_columns(df)
        
        # 2. 빈 값 확인
        self._check_empty_values(df)
        
        # 3. 라벨 유효성 확인
        self._check_label_validity(df)
        
        # 4. 데이터 규모 확인
        self._check_data_size(df)
        
        # 5. 라벨 분포 확인
        self._check_label_distribution(df)
        
        # 6. 세션 정보 확인 (있는 경우)
        if 'session_id' in df.columns:
            self._check_session_info(df)
        
        # 7. 텍스트 품질 확인
        self._check_text_quality(df)
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    
    def _check_required_columns(self, df: pd.DataFrame):
        """필수 컬럼 존재 확인"""
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            self.errors.append(f"필수 컬럼 누락: {', '.join(missing)}")
    
    def _check_empty_values(self, df: pd.DataFrame):
        """빈 값 확인"""
        if 'text' in df.columns:
            null_count = df['text'].isna().sum()
            if null_count > 0:
                self.errors.append(f"text 컬럼에 NULL 값 {null_count}개 존재")
            
            empty_count = (df['text'].astype(str).str.strip() == '').sum()
            if empty_count > 0:
                self.errors.append(f"text 컬럼에 빈 문자열 {empty_count}개 존재")
        
        if 'label' in df.columns:
            null_count = df['label'].isna().sum()
            if null_count > 0:
                self.errors.append(f"label 컬럼에 NULL 값 {null_count}개 존재")
    
    def _check_label_validity(self, df: pd.DataFrame):
        """라벨 유효성 확인"""
        if 'label' not in df.columns:
            return
        
        invalid_labels = []
        for idx, label_str in df['label'].items():
            if pd.isna(label_str):
                continue
            
            labels = str(label_str).split('|')
            for label in labels:
                label = label.strip()
                if label and label not in VALID_LABELS:
                    invalid_labels.append((idx, label))
        
        if invalid_labels:
            examples = invalid_labels[:5]  # 처음 5개만 표시
            self.errors.append(
                f"잘못된 라벨 {len(invalid_labels)}개 발견. 예시: {examples}"
            )
    
    def _check_data_size(self, df: pd.DataFrame):
        """데이터 규모 확인"""
        total = len(df)
        
        if total < 500:
            self.warnings.append(
                f"데이터가 너무 적음: {total}개 (최소 500개 권장)"
            )
        elif total < 1000:
            self.warnings.append(
                f"데이터 규모가 작음: {total}개 (1,000개 이상 권장)"
            )
    
    def _check_label_distribution(self, df: pd.DataFrame):
        """라벨 분포 확인"""
        if 'label' not in df.columns:
            return
        
        label_counts = {}
        for label_str in df['label']:
            if pd.isna(label_str):
                continue
            labels = str(label_str).split('|')
            for label in labels:
                label = label.strip()
                if label:
                    label_counts[label] = label_counts.get(label, 0) + 1
        
        # 각 라벨별 최소 개수 확인
        for label, count in label_counts.items():
            if count < 20:
                self.warnings.append(
                    f"라벨 '{label}' 샘플 부족: {count}개 (최소 20개 권장)"
                )
        
        # 정상 샘플 비율 확인
        normal_count = label_counts.get('정상', 0)
        total = len(df)
        if total > 0:
            normal_ratio = normal_count / total
            if normal_ratio < 0.3:
                self.warnings.append(
                    f"정상 샘플 비율이 낮음: {normal_ratio:.1%} (30% 이상 권장)"
                )
            elif normal_ratio > 0.8:
                self.warnings.append(
                    f"정상 샘플 비율이 너무 높음: {normal_ratio:.1%} (클래스 불균형 가능)"
                )
    
    def _check_session_info(self, df: pd.DataFrame):
        """세션 정보 확인"""
        if 'session_id' not in df.columns:
            return
        
        # 세션별 턴 수 확인
        if 'turn_id' in df.columns:
            session_turns = df.groupby('session_id')['turn_id'].count()
            single_turn_sessions = (session_turns == 1).sum()
            if single_turn_sessions > 0:
                self.warnings.append(
                    f"턴이 1개인 세션 {single_turn_sessions}개 (반복성 감지 어려움)"
                )
    
    def _check_text_quality(self, df: pd.DataFrame):
        """텍스트 품질 확인"""
        if 'text' not in df.columns:
            return
        
        # 텍스트 길이 확인
        text_lengths = df['text'].astype(str).str.len()
        too_short = (text_lengths < 3).sum()
        too_long = (text_lengths > 1000).sum()
        
        if too_short > 0:
            self.warnings.append(f"너무 짧은 텍스트 {too_short}개 (3자 미만)")
        if too_long > 0:
            self.warnings.append(f"너무 긴 텍스트 {too_long}개 (1000자 초과)")
        
        # 개인정보 패턴 확인 (간단한 체크)
        phone_pattern = r'\d{2,3}-\d{3,4}-\d{4}'
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        
        phone_count = df['text'].astype(str).str.contains(phone_pattern, regex=True).sum()
        email_count = df['text'].astype(str).str.contains(email_pattern, regex=True).sum()
        
        if phone_count > 0:
            self.warnings.append(
                f"전화번호 패턴 발견 {phone_count}개 (개인정보 마스킹 필요)"
            )
        if email_count > 0:
            self.warnings.append(
                f"이메일 패턴 발견 {email_count}개 (개인정보 마스킹 필요)"
            )


def load_dataset(file_path: str, format: Optional[str] = None) -> pd.DataFrame:
    """
    데이터셋 로드 (CSV, JSON, Excel 지원)
    
    Args:
        file_path: 파일 경로
        format: 파일 형식 ('csv', 'json', 'excel', None=자동 감지)
    
    Returns:
        DataFrame
    """
    path = Path(file_path)
    
    # 형식 자동 감지
    if format is None:
        if path.suffix.lower() == '.csv':
            format = 'csv'
        elif path.suffix.lower() in ['.json', '.jsonl']:
            format = 'json'
        elif path.suffix.lower() in ['.xlsx', '.xls']:
            format = 'excel'
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {path.suffix}")
    
    # 파일 로드
    if format == 'csv':
        # UTF-8 또는 UTF-8-sig 시도
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    elif format == 'json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # JSON 형식에 따라 변환
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and 'sessions' in data:
            # 세션 기반 JSON 형식
            rows = []
            for session in data['sessions']:
                session_id = session.get('session_id', '')
                for turn in session.get('turns', []):
                    row = {
                        'text': turn.get('text', ''),
                        'label': '|'.join(turn.get('labels', [])),
                        'session_id': session_id,
                        'turn_id': turn.get('turn_id', ''),
                        'speaker': turn.get('speaker', ''),
                        'severity': '|'.join(turn.get('severities', [])) if isinstance(turn.get('severities'), list) else turn.get('severity', '')
                    }
                    rows.append(row)
            df = pd.DataFrame(rows)
        else:
            df = pd.json_normalize(data)
    
    elif format == 'excel':
        df = pd.read_excel(file_path)
    
    else:
        raise ValueError(f"지원하지 않는 형식: {format}")
    
    return df


def validate_dataset(file_path: str, format: Optional[str] = None, 
                    required_columns: List[str] = None) -> Tuple[bool, List[str], List[str]]:
    """
    데이터셋 파일 검증
    
    Args:
        file_path: 파일 경로
        format: 파일 형식 (None=자동 감지)
        required_columns: 필수 컬럼 리스트
    
    Returns:
        (is_valid, errors, warnings)
    """
    # 데이터 로드
    try:
        df = load_dataset(file_path, format)
    except Exception as e:
        return False, [f"파일 로드 실패: {str(e)}"], []
    
    # 검증
    validator = DatasetValidator(required_columns)
    return validator.validate(df)


def print_validation_report(file_path: str, format: Optional[str] = None):
    """검증 결과를 보기 좋게 출력"""
    print("=" * 80)
    print(f"데이터셋 검증: {file_path}")
    print("=" * 80)
    
    is_valid, errors, warnings = validate_dataset(file_path, format)
    
    print(f"\n📊 검증 결과: {'✅ 통과' if is_valid else '❌ 실패'}")
    print(f"   - 전체 데이터: {len(load_dataset(file_path, format))}개")
    
    if errors:
        print(f"\n❌ 오류 ({len(errors)}개):")
        for i, error in enumerate(errors, 1):
            print(f"   {i}. {error}")
    
    if warnings:
        print(f"\n⚠️  경고 ({len(warnings)}개):")
        for i, warning in enumerate(warnings, 1):
            print(f"   {i}. {warning}")
    
    if not errors and not warnings:
        print("\n✅ 모든 검증 통과!")
    
    print("=" * 80)
    
    return is_valid


# 사용 예제
if __name__ == "__main__":
    # 예제: 데이터셋 검증
    # print_validation_report('data/train.csv')
    
    # 예제: 데이터셋 로드
    # df = load_dataset('data/train.csv')
    # print(df.head())
    
    print("데이터셋 검증 도구 준비 완료!")
    print("사용법: print_validation_report('your_dataset.csv')")


