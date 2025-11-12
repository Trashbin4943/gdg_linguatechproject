"""
빠른 시작 가이드

새로운 데이터셋을 준비한 후 바로 실행할 수 있는 스크립트
"""

import pandas as pd
from pathlib import Path
import sys
import os

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from risk_based_classifier import RiskScoreClassifier, ConsultationMetadata
from utils.dataset_validator import load_dataset, print_validation_report


def quick_start_with_new_dataset(dataset_path: str):
    """
    새로운 데이터셋으로 빠른 시작
    
    Args:
        dataset_path: 데이터셋 파일 경로 (CSV)
    """
    print("=" * 80)
    print("악성 민원 분류 시스템 - 빠른 시작")
    print("=" * 80)
    
    # 1. 데이터 검증
    print("\n[1단계] 데이터 검증")
    print("-" * 80)
    is_valid, errors, warnings = print_validation_report(dataset_path)
    
    if not is_valid:
        print("\n❌ 데이터 검증 실패. 오류를 수정한 후 다시 시도하세요.")
        return
    
    # 2. 데이터 로드
    print("\n[2단계] 데이터 로드")
    print("-" * 80)
    df = load_dataset(dataset_path)
    print(f"✅ 데이터 로드 완료: {len(df)}개 행")
    print(f"컬럼: {list(df.columns)}")
    
    # 3. 분류기 초기화
    print("\n[3단계] 분류기 초기화")
    print("-" * 80)
    classifier = RiskScoreClassifier()
    print("✅ 분류기 초기화 완료")
    
    # 4. 샘플 테스트
    print("\n[4단계] 샘플 테스트")
    print("-" * 80)
    
    # 샘플 데이터 선택 (최대 10개)
    sample_size = min(10, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    
    results_summary = {
        'NORMAL': 0,
        'LOW': 0,
        'MEDIUM': 0,
        'HIGH': 0,
        'CRITICAL': 0
    }
    
    for idx, row in sample_df.iterrows():
        text = str(row.get('text', ''))
        if not text or len(text.strip()) < 3:
            continue
        
        # 메타데이터 추출 (있는 경우)
        metadata = None
        if 'consultation_content' in df.columns or 'consultation_result' in df.columns:
            metadata = ConsultationMetadata(
                consultation_content=row.get('consultation_content'),
                consultation_result=row.get('consultation_result'),
                requirement_type=row.get('requirement_type'),
                consultation_reason=row.get('consultation_reason')
            )
        
        # 분류
        result = classifier.classify(text, metadata=metadata)
        
        # 결과 집계
        risk_level_name = result.risk_level.name
        results_summary[risk_level_name] = results_summary.get(risk_level_name, 0) + 1
        
        # 결과 출력
        print(f"\n[{idx}] {text[:50]}...")
        print(f"  위험도 점수: {result.risk_score}/10")
        print(f"  위험도 레벨: {risk_level_name}")
        if result.profanity_detected:
            print(f"  욕설 감지: ✅ ({result.profanity_category})")
        if result.baseline_issues:
            print(f"  Baseline 이슈: {', '.join(result.baseline_issues[:3])}")
        if result.metadata_issues:
            print(f"  메타데이터 이슈: {', '.join(result.metadata_issues[:3])}")
    
    # 5. 요약
    print("\n" + "=" * 80)
    print("[요약]")
    print("=" * 80)
    print(f"테스트 샘플 수: {sample_size}개")
    print(f"\n위험도 레벨 분포:")
    for level, count in results_summary.items():
        if count > 0:
            print(f"  {level}: {count}개 ({count/sample_size*100:.1f}%)")
    
    print("\n✅ 빠른 시작 완료!")
    print("\n다음 단계:")
    print("1. 전체 데이터셋으로 분류 실행")
    print("2. 결과 분석 및 개선")
    print("3. 필요시 모델 Fine-tuning")


def batch_classify_dataset(dataset_path: str, output_path: str = None):
    """
    전체 데이터셋 배치 분류
    
    Args:
        dataset_path: 입력 데이터셋 경로
        output_path: 출력 파일 경로 (None이면 자동 생성)
    """
    print("=" * 80)
    print("전체 데이터셋 배치 분류")
    print("=" * 80)
    
    # 데이터 로드
    df = load_dataset(dataset_path)
    print(f"\n데이터 로드: {len(df)}개 행")
    
    # 분류기 초기화
    classifier = RiskScoreClassifier()
    
    # 결과 저장용 리스트
    results_list = []
    
    # 배치 분류
    print("\n분류 진행 중...")
    for idx, row in df.iterrows():
        text = str(row.get('text', ''))
        if not text or len(text.strip()) < 3:
            continue
        
        # 메타데이터 추출
        metadata = None
        if 'consultation_content' in df.columns:
            metadata = ConsultationMetadata(
                consultation_content=row.get('consultation_content'),
                consultation_result=row.get('consultation_result'),
                requirement_type=row.get('requirement_type'),
                consultation_reason=row.get('consultation_reason')
            )
        
        # 분류
        result = classifier.classify(text, metadata=metadata)
        
        # 결과 저장
        results_list.append({
            'text': text,
            'risk_score': result.risk_score,
            'risk_level': result.risk_level.name,
            'profanity_detected': result.profanity_detected,
            'profanity_category': result.profanity_category,
            'baseline_issues': '|'.join(result.baseline_issues) if result.baseline_issues else '',
            'metadata_issues': '|'.join(result.metadata_issues) if result.metadata_issues else '',
            'confidence': result.confidence,
            'recommendation': result.recommendation,
            **{col: row.get(col, '') for col in df.columns if col != 'text'}
        })
        
        if (idx + 1) % 100 == 0:
            print(f"  진행: {idx + 1}/{len(df)}")
    
    # 결과 저장
    results_df = pd.DataFrame(results_list)
    
    if output_path is None:
        output_path = dataset_path.replace('.csv', '_classified.csv')
    
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 분류 완료: {len(results_list)}개")
    print(f"결과 저장: {output_path}")
    
    # 통계 출력
    print("\n위험도 레벨 분포:")
    level_counts = results_df['risk_level'].value_counts()
    for level, count in level_counts.items():
        print(f"  {level}: {count}개 ({count/len(results_df)*100:.1f}%)")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='악성 민원 분류 시스템 빠른 시작')
    parser.add_argument('dataset_path', type=str, help='데이터셋 파일 경로')
    parser.add_argument('--batch', action='store_true', help='전체 데이터셋 배치 분류')
    parser.add_argument('--output', type=str, default=None, help='출력 파일 경로')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_classify_dataset(args.dataset_path, args.output)
    else:
        quick_start_with_new_dataset(args.dataset_path)


