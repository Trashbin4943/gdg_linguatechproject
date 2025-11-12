"""
윤리검증 데이터셋을 악성 민원 형식으로 변환

텍스트 윤리검증 데이터 → 우리 시스템 형식 (CSV)
"""

import zipfile
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
import os

# 매핑 정의
ETHICS_TO_COMPLAINT_MAPPING = {
    "CENSURE": "모욕_조롱",
    "HATE": "혐오표현",
    "DISCRIMINATION": "혐오표현",
    "SEXUAL": "외설_성희롱",
    "ABUSE": "폭력_위협_범죄조장",
    "VIOLENCE": "폭력_위협_범죄조장",
    "CRIME": "폭력_위협_범죄조장",
}

def map_intensity_to_severity(intensity: float) -> str:
    """비윤리 강도를 심각도로 변환"""
    if intensity < 2.0:
        return "LOW"
    elif intensity == 2.0:
        return "MEDIUM"
    else:
        return "HIGH"

def convert_ethics_dataset(zip_path: str, output_path: str, split: str = "train"):
    """
    윤리검증 데이터셋을 악성 민원 형식으로 변환
    
    Args:
        zip_path: talksets ZIP 파일 경로
        output_path: 출력 CSV 파일 경로
        split: "train" 또는 "valid"
    """
    print("=" * 80)
    print(f"윤리검증 데이터셋 변환: {split}")
    print("=" * 80)
    
    if not os.path.exists(zip_path):
        print(f"❌ 파일을 찾을 수 없습니다: {zip_path}")
        return None
    
    z = zipfile.ZipFile(zip_path)
    files = [f for f in z.filelist if 'talksets' in f.filename and f.filename.endswith('.json')]
    
    print(f"\n발견된 talksets 파일: {len(files)}개")
    
    all_rows = []
    temp_dir = Path('temp_extract_ethics')
    temp_dir.mkdir(exist_ok=True)
    
    for file_info in files:
        print(f"\n처리 중: {file_info.filename}")
        
        # 압축 해제
        z.extract(file_info, temp_dir)
        json_path = temp_dir / file_info.filename
        
        # JSON 읽기
        with open(json_path, 'r', encoding='utf-8') as f:
            talksets = json.load(f)
        
        print(f"  talksets 수: {len(talksets)}")
        
        # 각 talkset 처리
        for talkset in talksets:
            talkset_id = talkset['id']
            
            # 각 sentence 처리
            for sentence in talkset.get('sentences', []):
                text = sentence.get('text', '').strip()
                if not text:
                    continue
                
                # 라벨 변환
                labels = []
                is_immoral = sentence.get('is_immoral', False)
                
                if is_immoral:
                    ethics_types = sentence.get('types', [])
                    for ethics_type in ethics_types:
                        if ethics_type in ETHICS_TO_COMPLAINT_MAPPING:
                            complaint_label = ETHICS_TO_COMPLAINT_MAPPING[ethics_type]
                            if complaint_label not in labels:
                                labels.append(complaint_label)
                else:
                    labels = ["정상"]
                
                # 심각도 변환
                intensity = sentence.get('intensity', 0)
                if is_immoral:
                    severity = map_intensity_to_severity(intensity)
                else:
                    severity = "NORMAL"
                
                # 행 추가
                row = {
                    'text': text,
                    'label': '|'.join(labels) if labels else '정상',
                    'severity': severity,
                    'session_id': talkset_id,
                    'turn_id': sentence.get('speaker', 1),
                    'intensity': intensity,
                    'is_immoral': is_immoral,
                    'ethics_types': '|'.join(sentence.get('types', [])),
                    'speaker': sentence.get('speaker', 1),
                    'split': split
                }
                all_rows.append(row)
        
        print(f"  처리된 문장: {len([r for r in all_rows if r['split'] == split])}개")
    
    # DataFrame 생성
    df = pd.DataFrame(all_rows)
    
    # CSV 저장
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*80}")
    print("변환 완료")
    print(f"{'='*80}")
    print(f"총 문장 수: {len(df)}개")
    print(f"저장 위치: {output_path}")
    
    # 통계 출력
    print(f"\n라벨 분포:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count}개 ({count/len(df)*100:.1f}%)")
    
    print(f"\n심각도 분포:")
    severity_counts = df['severity'].value_counts()
    for severity, count in severity_counts.items():
        print(f"  {severity}: {count}개 ({count/len(df)*100:.1f}%)")
    
    print(f"\n비윤리 문장: {df['is_immoral'].sum()}개 ({df['is_immoral'].sum()/len(df)*100:.1f}%)")
    print(f"정상 문장: {(~df['is_immoral']).sum()}개 ({(~df['is_immoral']).sum()/len(df)*100:.1f}%)")
    
    z.close()
    return df


def convert_all_datasets(base_path: str, output_dir: str = "converted_data"):
    """
    Training과 Validation 데이터 모두 변환
    
    Args:
        base_path: 데이터셋 기본 경로
        output_dir: 출력 디렉토리
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Training 데이터 변환
    train_zip = os.path.join(base_path, "1.Training", "라벨링데이터", "aihub", "TL1_aihub.zip")
    train_output = output_path / "ethics_train.csv"
    
    if os.path.exists(train_zip):
        print("\n" + "="*80)
        print("TRAINING 데이터 변환")
        print("="*80)
        train_df = convert_ethics_dataset(train_zip, str(train_output), "train")
    else:
        print(f"❌ Training 파일을 찾을 수 없습니다: {train_zip}")
        train_df = None
    
    # Validation 데이터 변환
    val_zip = os.path.join(base_path, "2.Validation", "라벨링데이터", "aihub", "VL1_aihub.zip")
    val_output = output_path / "ethics_valid.csv"
    
    if os.path.exists(val_zip):
        print("\n" + "="*80)
        print("VALIDATION 데이터 변환")
        print("="*80)
        val_df = convert_ethics_dataset(val_zip, str(val_output), "valid")
    else:
        print(f"❌ Validation 파일을 찾을 수 없습니다: {val_zip}")
        val_df = None
    
    return train_df, val_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='윤리검증 데이터셋 변환')
    parser.add_argument('--base_path', type=str, 
                       default=r'C:\Users\SAMSUNG\Downloads\147.텍스트 윤리검증 데이터\01.데이터',
                       help='데이터셋 기본 경로')
    parser.add_argument('--output_dir', type=str, default='converted_data',
                       help='출력 디렉토리')
    
    args = parser.parse_args()
    
    # 전체 변환
    train_df, val_df = convert_all_datasets(args.base_path, args.output_dir)
    
    print("\n" + "="*80)
    print("변환 완료!")
    print("="*80)
    print(f"\n다음 단계:")
    print("1. 변환된 데이터 검증: dataset_validator.py 사용")
    print("2. Baseline 테스트: quick_start.py 사용")
    print("3. 모델 학습: kobert_finetuning.py 사용")


