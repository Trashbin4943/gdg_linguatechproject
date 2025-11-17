"""
Risk Score 기반 분류 시스템

2단계 파이프라인:
1. 욕설 필터링 (전용 모델) → 직접적 악성 민원 분리
2. Risk Score 기반 분류 → 간접적 악성 민원 위험도 평가
"""

from enum import Enum
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from classification_criteria import (
    ClassificationCriteria,
    ClassificationResult,
    ComplaintCategory,
    ComplaintSeverity
)


class RiskLevel(Enum):
    """위험도 레벨"""
    NORMAL = 0      # 정상 (0-2점)
    LOW = 1         # 낮은 위험 (3-4점)
    MEDIUM = 2      # 보통 위험 (5-6점)
    HIGH = 3        # 높은 위험 (7-8점)
    CRITICAL = 4    # 심각 (9-10점)


@dataclass
class ConsultationMetadata:
    """상담 메타데이터"""
    consultation_content: Optional[str] = None      # 상담 내용
    consultation_result: Optional[str] = None       # 상담 결과
    requirement_type: Optional[str] = None          # 상담 요건
    consultation_reason: Optional[str] = None       # 상담 사유
    category: Optional[str] = None                  # 카테고리
    subcategory: Optional[str] = None               # 서브카테고리


@dataclass
class RiskScoreResult:
    """Risk Score 결과"""
    risk_score: int                    # 위험도 점수 (0-10)
    risk_level: RiskLevel              # 위험도 레벨
    profanity_detected: bool           # 욕설 감지 여부
    profanity_category: Optional[str]  # 욕설 카테고리
    baseline_issues: List[str]         # Baseline 감지 이슈
    metadata_issues: List[str]         # 메타데이터 기반 이슈
    confidence: float                  # 신뢰도 (0.0-1.0)
    recommendation: str                # 권장 조치


class ProfanityFilter:
    """욕설 필터링 전용 모델"""
    
    def __init__(self, use_korcen: bool = True, model_name: str = 'monologg/kobert'):
        """
        Args:
            use_korcen: korcen.py 기반 필터 사용 여부 (기본: True)
            model_name: 사전 학습된 모델 (욕설 분류용으로 Fine-tuning 필요, 현재 미사용)
        """
        self.use_korcen = False
        self.use_baseline = True  # 기본값: Baseline 규칙 사용
        
        if use_korcen:
            # 통화 상담 특화 필터 사용
            try:
                from call_center_profanity_filter import CallCenterProfanityFilter
                self.korcen_filter = CallCenterProfanityFilter()
                self.use_korcen = True
                self.use_baseline = False
            except ImportError:
                print("Warning: call_center_profanity_filter를 로드할 수 없습니다. Baseline 규칙을 사용합니다.")
                self.use_korcen = False
                self.use_baseline = True
    
    def detect_profanity(self, text: str) -> Tuple[bool, Optional[str], float]:
        """
        욕설 감지
        
        Returns:
            (is_profanity, category, confidence)
        """
        if self.use_korcen and hasattr(self, 'korcen_filter'):
            # korcen.py 기반 통화 상담 특화 필터 사용
            try:
                # check_profanity의 반환값 형식에 따라 처리
                profanity_result = self.korcen_filter.check_profanity(text)
                
                # 반환값이 튜플인 경우 처리
                if isinstance(profanity_result, tuple):
                    if len(profanity_result) >= 3:
                        is_prof, level, pattern = profanity_result[:3]
                        confidence = profanity_result[3] if len(profanity_result) > 3 else 0.8
                    else:
                        is_prof, level = profanity_result[:2]
                        pattern = None
                        confidence = 0.8
                else:
                    # 단일 값 반환인 경우
                    is_prof = bool(profanity_result)
                    level = None
                    pattern = None
                    confidence = 0.8
                
                if is_prof and level:
                    if hasattr(self.korcen_filter, 'get_profanity_category'):
                        category = self.korcen_filter.get_profanity_category(level)
                    else:
                        category = str(level)
                    return True, category, confidence
                
                return False, None, 0.0
            except Exception as e:
                print(f"Warning: korcen 필터 실행 중 오류 발생: {e}. Baseline 규칙으로 전환합니다.")
                self.use_korcen = False
                self.use_baseline = True
        
        if self.use_baseline:
            # Baseline 규칙으로 욕설 감지
            results = ClassificationCriteria.classify_text(text)
            
            profanity_categories = [
                ComplaintCategory.PROFANITY,
                ComplaintCategory.INSULT,
                ComplaintCategory.VIOLENCE_THREAT,
                ComplaintCategory.SEXUAL_HARASSMENT
            ]
            
            for result in results:
                if result.category in profanity_categories:
                    return True, result.category.value, result.confidence
            
            return False, None, 0.0
        
        # 기본값: 욕설 없음
        return False, None, 0.0
    
    def filter_profanity(self, text: str) -> Optional[RiskScoreResult]:
        """
        욕설 필터링 - 직접적 악성 민원 분리
        
        Returns:
            RiskScoreResult (욕설 감지된 경우) 또는 None
        """
        is_profanity, category, confidence = self.detect_profanity(text)
        
        if is_profanity:
            # 욕설 감지 → 즉시 CRITICAL 처리
            return RiskScoreResult(
                risk_score=10,
                risk_level=RiskLevel.CRITICAL,
                profanity_detected=True,
                profanity_category=category,
                baseline_issues=[category],
                metadata_issues=[],
                confidence=confidence,
                recommendation="즉시 조치 필요: 법적 조치 고려 (녹음 보관, 고소 검토)"
            )
        
        return None


class RiskScoreClassifier:
    """Risk Score 기반 분류기"""
    
    def __init__(self):
        self.profanity_filter = ProfanityFilter()
    
    def calculate_baseline_risk(self, text: str, session_context: Optional[List[str]] = None) -> Tuple[int, List[str]]:
        """
        Baseline 규칙 기반 위험도 점수 계산
        
        Returns:
            (risk_score, issues)
        """
        results = ClassificationCriteria.classify_text(text, session_context)
        
        risk_score = 0
        issues = []
        
        for result in results:
            if result.severity == ComplaintSeverity.NORMAL:
                continue
            
            # 심각도에 따른 점수
            if result.severity == ComplaintSeverity.CRITICAL:
                risk_score += 4
            elif result.severity == ComplaintSeverity.HIGH:
                risk_score += 3
            elif result.severity == ComplaintSeverity.MEDIUM:
                risk_score += 2
            elif result.severity == ComplaintSeverity.LOW:
                risk_score += 1
            
            issues.append(f"{result.category.value} ({result.severity.value})")
        
        return min(risk_score, 10), issues
    
    def calculate_metadata_risk(self, metadata: Optional[ConsultationMetadata]) -> Tuple[int, List[str]]:
        """
        메타데이터 기반 위험도 점수 계산
        
        Returns:
            (risk_score, issues)
        """
        if not metadata:
            return 0, []
        
        risk_score = 0
        issues = []
        
        # 상담 내용 분석
        content = metadata.consultation_content
        if content == '고충 상담':
            risk_score += 3
            issues.append('고충 상담')
        elif content == '업무 처리':
            risk_score += 1
        
        # 상담 결과 분석
        result = metadata.consultation_result
        if result == '해결 불가':
            risk_score += 3
            issues.append('해결 불가')
        elif result == '미흡':
            risk_score += 2
            issues.append('미흡')
        elif result == '추가 상담 필요':
            risk_score += 1
            issues.append('추가 상담 필요')
        
        # 상담 요건
        requirement = metadata.requirement_type
        if requirement == '다수 요건':
            risk_score += 1
            issues.append('다수 요건 (반복성 의심)')
        
        # 상담 사유
        reason = metadata.consultation_reason
        if reason == '업체':
            risk_score += 1
            issues.append('업체 사유')
        
        return min(risk_score, 10), issues
    
    def classify(
        self,
        text: str,
        session_context: Optional[List[str]] = None,
        metadata: Optional[ConsultationMetadata] = None
    ) -> RiskScoreResult:
        """
        Risk Score 기반 분류
        
        Returns:
            RiskScoreResult
        """
        # 1단계: 욕설 필터링 (직접적 악성 민원)
        profanity_result = self.profanity_filter.filter_profanity(text)
        if profanity_result:
            return profanity_result
        
        # 2단계: Risk Score 계산 (간접적 악성 민원)
        baseline_score, baseline_issues = self.calculate_baseline_risk(text, session_context)
        metadata_score, metadata_issues = self.calculate_metadata_risk(metadata)
        
        # 통합 Risk Score (가중 평균)
        total_score = max(baseline_score, metadata_score)  # 더 높은 점수 사용
        
        # 위험도 레벨 결정
        if total_score >= 9:
            risk_level = RiskLevel.CRITICAL
        elif total_score >= 7:
            risk_level = RiskLevel.HIGH
        elif total_score >= 5:
            risk_level = RiskLevel.MEDIUM
        elif total_score >= 3:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.NORMAL
        
        # 신뢰도 계산
        confidence = min(0.5 + (total_score / 20), 1.0)
        
        # 권장 조치
        recommendation = self._get_recommendation(risk_level, total_score)
        
        return RiskScoreResult(
            risk_score=total_score,
            risk_level=risk_level,
            profanity_detected=False,
            profanity_category=None,
            baseline_issues=baseline_issues,
            metadata_issues=metadata_issues,
            confidence=confidence,
            recommendation=recommendation
        )
    
    def _get_recommendation(self, risk_level: RiskLevel, risk_score: int) -> str:
        """위험도에 따른 권장 조치"""
        recommendations = {
            RiskLevel.NORMAL: "정상 처리",
            RiskLevel.LOW: "모니터링",
            RiskLevel.MEDIUM: "경고 안내",
            RiskLevel.HIGH: "특별 관리 대상 등록",
            RiskLevel.CRITICAL: "법적 조치 고려 (녹음 보관, 고소 검토)"
        }
        return recommendations.get(risk_level, "확인 필요")
    
    def batch_classify(
        self,
        texts: List[str],
        session_contexts: Optional[List[List[str]]] = None,
        metadata_list: Optional[List[ConsultationMetadata]] = None
    ) -> List[RiskScoreResult]:
        """배치 분류"""
        results = []
        
        for i, text in enumerate(texts):
            context = session_contexts[i] if session_contexts else None
            metadata = metadata_list[i] if metadata_list else None
            
            result = self.classify(text, context, metadata)
            results.append(result)
        
        return results


# 사용 예제
if __name__ == "__main__":
    classifier = RiskScoreClassifier()
    
    # 예제 1: 욕설 감지
    text1 = "너 같은 새끼는 진작에 죽었어야 해!"
    result1 = classifier.classify(text1)
    print("=" * 80)
    print("예제 1: 욕설 감지")
    print("=" * 80)
    print(f"텍스트: {text1}")
    print(f"위험도 점수: {result1.risk_score}/10")
    print(f"위험도 레벨: {result1.risk_level.name}")
    print(f"욕설 감지: {result1.profanity_detected}")
    print(f"욕설 카테고리: {result1.profanity_category}")
    print(f"권장 조치: {result1.recommendation}")
    
    # 예제 2: 메타데이터 기반 위험도
    text2 = "그렇군요"
    metadata2 = ConsultationMetadata(
        consultation_content='고충 상담',
        consultation_result='해결 불가',
        requirement_type='다수 요건'
    )
    result2 = classifier.classify(text2, metadata=metadata2)
    print("\n" + "=" * 80)
    print("예제 2: 메타데이터 기반 위험도")
    print("=" * 80)
    print(f"텍스트: {text2}")
    print(f"위험도 점수: {result2.risk_score}/10")
    print(f"위험도 레벨: {result2.risk_level.name}")
    print(f"Baseline 이슈: {result2.baseline_issues}")
    print(f"메타데이터 이슈: {result2.metadata_issues}")
    print(f"권장 조치: {result2.recommendation}")
    
    # 예제 3: 정상 케이스
    text3 = "아니 뭐 언제까지 기다리라고요."
    metadata3 = ConsultationMetadata(
        consultation_content='일반 문의',
        consultation_result='만족',
        requirement_type='단일 요건'
    )
    result3 = classifier.classify(text3, metadata=metadata3)
    print("\n" + "=" * 80)
    print("예제 3: 정상 케이스")
    print("=" * 80)
    print(f"텍스트: {text3}")
    print(f"위험도 점수: {result3.risk_score}/10")
    print(f"위험도 레벨: {result3.risk_level.name}")
    print(f"권장 조치: {result3.recommendation}")

