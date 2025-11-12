"""
분류 기준 테스트 및 예제 코드
"""

from classification_criteria import (
    ClassificationCriteria, 
    ComplaintCategory, 
    ComplaintSeverity
)


def test_classification():
    """분류 기준 테스트"""
    
    # 테스트 케이스들
    test_cases = [
        {
            "text": "X팔 너 거기 앉아서 뭐 배웠느냐? 고등학교는 나왔느냐?",
            "context": [],
            "expected": [ComplaintCategory.PROFANITY, ComplaintCategory.INSULT]
        },
        {
            "text": "앞선 통화에서도 말씀드렸다시피, 또 같은 얘기인데요.",
            "context": ["이전 대화 내용 1", "이전 대화 내용 2"],
            "expected": [ComplaintCategory.REPETITION]
        },
        {
            "text": "죽여버리겠다. 찾아가서 복수하겠어.",
            "context": [],
            "expected": [ComplaintCategory.VIOLENCE_THREAT]
        },
        {
            "text": "공짜로 독도에 보내달라. 돈이 없는데 특별히 해줘.",
            "context": [],
            "expected": [ComplaintCategory.UNREASONABLE_DEMAND, ComplaintCategory.IRRELEVANCE]
        },
        {
            "text": "이건 장난이에요. 테스트로 전화한 거예요.",
            "context": [],
            "expected": [ComplaintCategory.PRANK_CALL]
        },
        {
            "text": "정상적인 민원 문의입니다. 도움이 필요합니다.",
            "context": [],
            "expected": []  # 정상
        }
    ]
    
    print("=" * 80)
    print("악성 민원 분류 테스트")
    print("=" * 80)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n[테스트 케이스 {i}]")
        print(f"입력 텍스트: {case['text']}")
        
        results = ClassificationCriteria.classify_text(
            case['text'], 
            case.get('context', None)
        )
        
        print(f"\n분류 결과:")
        for result in results:
            if result.severity != ComplaintSeverity.NORMAL:
                print(f"  - 카테고리: {result.category.value}")
                print(f"    심각도: {result.severity.value}")
                print(f"    신뢰도: {result.confidence:.2f}")
                print(f"    설명: {result.description}")
                if result.evidence:
                    print(f"    근거: {', '.join(result.evidence[:3])}...")
                print(f"    조치: {ClassificationCriteria.get_severity_action(result.severity)}")
        
        # 예상 결과와 비교
        detected_categories = {r.category for r in results if r.severity != ComplaintSeverity.NORMAL}
        expected_categories = set(case['expected'])
        
        if detected_categories == expected_categories or (not expected_categories and all(r.severity == ComplaintSeverity.NORMAL for r in results)):
            print(f"  ✅ 테스트 통과")
        else:
            print(f"  ⚠️  예상: {[c.value for c in expected_categories]}, 실제: {[c.value for c in detected_categories]}")
        
        print("-" * 80)


def classify_session(session_texts: list):
    """세션 전체를 분석하여 악성 민원 여부 판단"""
    print("\n" + "=" * 80)
    print("세션 전체 분석")
    print("=" * 80)
    
    all_results = []
    session_context = []
    
    for i, text in enumerate(session_texts, 1):
        print(f"\n[턴 {i}] {text[:50]}...")
        
        results = ClassificationCriteria.classify_text(text, session_context)
        session_context.append(text)  # 맥락에 추가
        
        # 정상이 아닌 결과만 수집
        non_normal = [r for r in results if r.severity != ComplaintSeverity.NORMAL]
        all_results.extend(non_normal)
        
        if non_normal:
            for result in non_normal:
                print(f"  ⚠️  {result.category.value} ({result.severity.value})")
    
    # 세션 전체 요약
    if all_results:
        print("\n" + "=" * 80)
        print("세션 요약")
        print("=" * 80)
        
        # 카테고리별 집계
        category_counts = {}
        max_severity = ComplaintSeverity.NORMAL
        
        for result in all_results:
            cat = result.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
            if result.severity.value > max_severity.value:
                max_severity = result.severity
        
        print(f"발견된 문제 카테고리: {len(category_counts)}개")
        for cat, count in category_counts.items():
            print(f"  - {cat}: {count}건")
        
        print(f"\n최고 심각도: {max_severity.value}")
        print(f"권장 조치: {ClassificationCriteria.get_severity_action(max_severity)}")
    else:
        print("\n✅ 정상 세션으로 판단됩니다.")


if __name__ == "__main__":
    # 단일 텍스트 테스트
    test_classification()
    
    # 세션 전체 테스트
    sample_session = [
        "안녕하세요, 문의가 있어서 전화드렸습니다.",
        "앞선 통화에서도 말씀드렸다시피, 같은 문제인데요.",
        "또 같은 얘기인데 왜 안 해주는 거예요?",
        "X팔, 제대로 일을 못하시나봐요?",
        "죽여버리겠어요, 정말 화가 나네요."
    ]
    
    classify_session(sample_session)


