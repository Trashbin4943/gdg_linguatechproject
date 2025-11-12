"""
코랩에서 바로 사용할 수 있는 분류 기준 사용 예제

사용법:
1. 이 파일을 코랩에 업로드
2. classification_criteria.py도 함께 업로드
3. 아래 코드 실행
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification_criteria import ClassificationCriteria, ComplaintCategory, ComplaintSeverity


def analyze_complaint_text(text: str, session_history: list = None):
    """
    민원 텍스트를 분석하여 악성 민원 여부 판단
    
    Args:
        text: 분석할 텍스트
        session_history: 이전 대화 내역 (반복성 감지용, 선택사항)
    
    Returns:
        dict: 분류 결과
    """
    results = ClassificationCriteria.classify_text(text, session_history)
    
    # 정상이 아닌 결과만 필터링
    issues = [r for r in results if r.severity != ComplaintSeverity.NORMAL]
    
    if not issues:
        return {
            "status": "정상",
            "severity": "NORMAL",
            "message": "정상 민원으로 판단됩니다."
        }
    
    # 가장 심각한 이슈 찾기
    max_severity = max(issues, key=lambda x: x.severity.value)
    
    return {
        "status": "문제 감지",
        "severity": max_severity.severity.value,
        "categories": [r.category.value for r in issues],
        "details": [
            {
                "category": r.category.value,
                "severity": r.severity.value,
                "confidence": round(r.confidence, 2),
                "description": r.description,
                "evidence": r.evidence[:3]  # 상위 3개만
            }
            for r in issues
        ],
        "recommended_action": ClassificationCriteria.get_severity_action(max_severity)
    }


# 사용 예제
if __name__ == "__main__":
    # 예제 1: 욕설 포함 텍스트
    text1 = "X팔 너 거기 앉아서 뭐 배웠느냐? 제대로 일을 못하시나봐요?"
    result1 = analyze_complaint_text(text1)
    print("예제 1:", text1)
    print("결과:", result1)
    print()
    
    # 예제 2: 반복성 감지 (세션 히스토리 포함)
    session = [
        "이전에 전화드렸는데 같은 문제로 또 문의드립니다.",
        "네, 확인해보겠습니다."
    ]
    text2 = "앞선 통화에서도 말씀드렸다시피, 또 같은 얘기인데요."
    result2 = analyze_complaint_text(text2, session)
    print("예제 2:", text2)
    print("결과:", result2)
    print()
    
    # 예제 3: 위협 표현
    text3 = "죽여버리겠다. 찾아가서 복수하겠어."
    result3 = analyze_complaint_text(text3)
    print("예제 3:", text3)
    print("결과:", result3)
    print()
    
    # 예제 4: 정상 민원
    text4 = "안녕하세요, 문의사항이 있어서 전화드렸습니다."
    result4 = analyze_complaint_text(text4)
    print("예제 4:", text4)
    print("결과:", result4)


