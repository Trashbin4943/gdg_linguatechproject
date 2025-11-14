"""
음성 상담에서 문제의 소지가 있을 수 있는 경우를 처리하는 분류 기준 정의

공공 상담과 민간 상담의 특성을 고려한 악성 민원 분류 체계
"""

from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass


class ComplaintSeverity(Enum):
    """민원 심각도 레벨"""
    NORMAL = "정상"  # 정상 민원
    LOW = "낮음"  # 경미한 문제
    MEDIUM = "보통"  # 주의 필요
    HIGH = "높음"  # 즉시 조치 필요
    CRITICAL = "심각"  # 법적 조치 고려


class ComplaintCategory(Enum):
    """악성 민원 분류 카테고리"""
    # 공공 상담 중심 (직접적 폭력)
    PROFANITY = "욕설_저주"  # 욕설, 저주
    INSULT = "모욕_조롱"  # 모욕, 조롱 (억측, 음모, 비하 포함)
    VIOLENCE_THREAT = "폭력_위협_범죄조장"  # 폭력, 위협, 범죄 조장
    SEXUAL_HARASSMENT = "외설_성희롱"  # 외설, 성희롱
    HATE_SPEECH = "혐오표현"  # 성/연령/인종/지역/장애/종교/정치/직업 혐오
    
    # 민간 상담 중심 (논리적 모호성)
    REPETITION = "반복성"  # 대화 진전 없음, 반복 민원
    UNREASONABLE_DEMAND = "무리한_요구"  # 매뉴얼/권한 밖 요구
    IRRELEVANCE = "부당성"  # 상담 맥락 이탈
    FALSE_COMPLAINT = "허위_민원"  # 허위 신고, 업무방해
    PRANK_CALL = "장난전화"  # 업무 무관 장난전화
    
    # 기타
    FEAR_INDUCING = "공포심_불안감_유발"  # 공포심/불안감 유발 행위


@dataclass
class ClassificationResult:
    """분류 결과 데이터 클래스"""
    category: ComplaintCategory
    severity: ComplaintSeverity
    confidence: float  # 0.0 ~ 1.0
    evidence: List[str]  # 판단 근거 (키워드, 문장 등)
    description: str  # 상세 설명


class ClassificationCriteria:
    """분류 기준 정의 및 판단 로직"""
    
    # 공공 상담 중심: 직접적 폭력 관련 키워드
    PROFANITY_KEYWORDS = [
        "X팔", "XXX년", "개XX", "XX놈", "XX년", "지랄", "병신", "미친",
        "씨발", "좆", "개새끼", "미친놈", "죽어", "꺼져"
    ]
    
    INSULT_KEYWORDS = [
        "너 거기 앉아서 뭐 배웠느냐", "고등학교는 나왔느냐", "인격모독",
        "바보", "멍청이", "무식한", "능력없는", "제대로 배우지 못한"
    ]
    
    THREAT_KEYWORDS = [
        "죽여버리겠다", "찾아가겠다", "법적 대응", "고소하겠다", "복수",
        "너희 다 죽어", "끝장내겠다", "망하게 하겠다"
    ]
    
    SEXUAL_HARASSMENT_KEYWORDS = [
        "성적인", "음란", "만나자", "연락처", "사적인", "데이트",
        "섹스", "성교", "음란물"
    ]
    
    # 민간 상담 중심: 반복성 감지 키워드
    REPETITION_INDICATORS = [
        "앞선 통화에서도 말씀드렸다시피", "이전에도 말씀드렸는데",
        "또 같은 말씀", "계속 같은 얘기", "반복해서 말씀드리는데",
        "또 물어보는 거예요", "아까도 말했는데"
    ]
    
    # 무리한 요구 감지 키워드
    UNREASONABLE_DEMAND_INDICATORS = [
        "공짜로", "무료로", "특별히", "예외로", "빠르게 해줘",
        "지금 당장", "불가능한데", "권한 밖", "할 수 없는데"
    ]
    
    # 부당성/무관성 감지 키워드
    IRRELEVANCE_INDICATORS = [
        "독도에 보내달라", "돈이 없는데", "상관없는 얘기",
        "이건 왜 물어보는 거예요", "맥락 없음"
    ]
    
    # 허위 민원 감지 키워드
    FALSE_COMPLAINT_INDICATORS = [
        "거짓말", "허위", "없는 일", "꾸며낸", "장난"
    ]
    
    # 장난전화 감지 키워드
    PRANK_CALL_INDICATORS = [
        "장난", "놀리는", "테스트", "연습", "이유리 상담원 찾아요"
    ]
    
    @staticmethod
    def classify_text(text: str, session_context: Optional[List[str]] = None) -> List[ClassificationResult]:
        """
        텍스트를 분석하여 악성 민원 분류 결과 반환
        
        Args:
            text: 분석할 텍스트
            session_context: 세션 내 이전 대화 맥락 (반복성 감지용)
        
        Returns:
            ClassificationResult 리스트 (여러 카테고리 동시 감지 가능)
        """
        results = []
        text_lower = text.lower()
        
        # 1. 욕설/저주 감지
        profanity_count = sum(1 for kw in ClassificationCriteria.PROFANITY_KEYWORDS if kw in text)
        if profanity_count > 0:
            results.append(ClassificationResult(
                category=ComplaintCategory.PROFANITY,
                severity=ComplaintSeverity.HIGH if profanity_count >= 3 else ComplaintSeverity.MEDIUM,
                confidence=min(0.5 + profanity_count * 0.15, 1.0),
                evidence=[kw for kw in ClassificationCriteria.PROFANITY_KEYWORDS if kw in text],
                description=f"욕설/저주 표현 {profanity_count}건 감지"
            ))
        
        # 2. 모욕/조롱 감지
        insult_count = sum(1 for kw in ClassificationCriteria.INSULT_KEYWORDS if kw in text)
        if insult_count > 0:
            results.append(ClassificationResult(
                category=ComplaintCategory.INSULT,
                severity=ComplaintSeverity.MEDIUM if insult_count >= 2 else ComplaintSeverity.LOW,
                confidence=min(0.4 + insult_count * 0.2, 1.0),
                evidence=[kw for kw in ClassificationCriteria.INSULT_KEYWORDS if kw in text],
                description=f"모욕/조롱 표현 {insult_count}건 감지"
            ))
        
        # 3. 폭력/위협 감지
        threat_count = sum(1 for kw in ClassificationCriteria.THREAT_KEYWORDS if kw in text)
        if threat_count > 0:
            results.append(ClassificationResult(
                category=ComplaintCategory.VIOLENCE_THREAT,
                severity=ComplaintSeverity.CRITICAL,
                confidence=min(0.7 + threat_count * 0.15, 1.0),
                evidence=[kw for kw in ClassificationCriteria.THREAT_KEYWORDS if kw in text],
                description=f"폭력/위협 표현 {threat_count}건 감지 - 즉시 조치 필요"
            ))
        
        # 4. 성희롱 감지
        sexual_count = sum(1 for kw in ClassificationCriteria.SEXUAL_HARASSMENT_KEYWORDS if kw in text)
        if sexual_count > 0:
            results.append(ClassificationResult(
                category=ComplaintCategory.SEXUAL_HARASSMENT,
                severity=ComplaintSeverity.CRITICAL,
                confidence=min(0.6 + sexual_count * 0.2, 1.0),
                evidence=[kw for kw in ClassificationCriteria.SEXUAL_HARASSMENT_KEYWORDS if kw in text],
                description=f"성희롱 표현 {sexual_count}건 감지 - 법적 조치 고려"
            ))
        
        # 5. 반복성 감지 (세션 맥락 필요)
        if session_context:
            repetition_count = sum(1 for indicator in ClassificationCriteria.REPETITION_INDICATORS 
                                  if indicator in text)
            # 이전 대화와의 유사도도 체크 (간단한 키워드 기반)
            similar_topics = sum(1 for prev_text in session_context[-3:] 
                               if any(word in prev_text and word in text 
                                     for word in text.split() if len(word) > 3))
            
            if repetition_count > 0 or similar_topics >= 2:
                results.append(ClassificationResult(
                    category=ComplaintCategory.REPETITION,
                    severity=ComplaintSeverity.MEDIUM if similar_topics >= 3 else ComplaintSeverity.LOW,
                    confidence=min(0.5 + (repetition_count + similar_topics) * 0.15, 1.0),
                    evidence=[indicator for indicator in ClassificationCriteria.REPETITION_INDICATORS 
                             if indicator in text],
                    description=f"반복성 감지: 반복 표현 {repetition_count}건, 유사 주제 {similar_topics}건"
                ))
        
        # 6. 무리한 요구 감지
        unreasonable_count = sum(1 for kw in ClassificationCriteria.UNREASONABLE_DEMAND_INDICATORS 
                                if kw in text)
        if unreasonable_count >= 2:
            results.append(ClassificationResult(
                category=ComplaintCategory.UNREASONABLE_DEMAND,
                severity=ComplaintSeverity.MEDIUM,
                confidence=min(0.4 + unreasonable_count * 0.2, 1.0),
                evidence=[kw for kw in ClassificationCriteria.UNREASONABLE_DEMAND_INDICATORS if kw in text],
                description=f"무리한 요구 표현 {unreasonable_count}건 감지"
            ))
        
        # 7. 부당성/무관성 감지
        irrelevance_count = sum(1 for kw in ClassificationCriteria.IRRELEVANCE_INDICATORS if kw in text)
        if irrelevance_count > 0:
            results.append(ClassificationResult(
                category=ComplaintCategory.IRRELEVANCE,
                severity=ComplaintSeverity.LOW,
                confidence=min(0.3 + irrelevance_count * 0.25, 1.0),
                evidence=[kw for kw in ClassificationCriteria.IRRELEVANCE_INDICATORS if kw in text],
                description=f"상담 맥락 이탈 표현 {irrelevance_count}건 감지"
            ))
        
        # 8. 허위 민원 감지
        false_count = sum(1 for kw in ClassificationCriteria.FALSE_COMPLAINT_INDICATORS if kw in text)
        if false_count > 0:
            results.append(ClassificationResult(
                category=ComplaintCategory.FALSE_COMPLAINT,
                severity=ComplaintSeverity.HIGH,
                confidence=min(0.5 + false_count * 0.25, 1.0),
                evidence=[kw for kw in ClassificationCriteria.FALSE_COMPLAINT_INDICATORS if kw in text],
                description=f"허위 민원 의심 표현 {false_count}건 감지"
            ))
        
        # 9. 장난전화 감지
        prank_count = sum(1 for kw in ClassificationCriteria.PRANK_CALL_INDICATORS if kw in text)
        if prank_count > 0:
            results.append(ClassificationResult(
                category=ComplaintCategory.PRANK_CALL,
                severity=ComplaintSeverity.MEDIUM,
                confidence=min(0.5 + prank_count * 0.2, 1.0),
                evidence=[kw for kw in ClassificationCriteria.PRANK_CALL_INDICATORS if kw in text],
                description=f"장난전화 의심 표현 {prank_count}건 감지"
            ))
        
        # 결과가 없으면 정상 민원
        if not results:
            results.append(ClassificationResult(
                category=ComplaintCategory.REPETITION,  # 임시 (정상 카테고리 추가 필요)
                severity=ComplaintSeverity.NORMAL,
                confidence=0.9,
                evidence=[],
                description="정상 민원으로 판단"
            ))
        
        return results
    
    @staticmethod
    def get_severity_action(severity: ComplaintSeverity) -> str:
        """심각도에 따른 조치 방안 반환"""
        actions = {
            ComplaintSeverity.NORMAL: "정상 처리",
            ComplaintSeverity.LOW: "모니터링",
            ComplaintSeverity.MEDIUM: "경고 안내",
            ComplaintSeverity.HIGH: "특별 관리 대상 등록",
            ComplaintSeverity.CRITICAL: "법적 조치 고려 (녹음 보관, 고소 검토)"
        }
        return actions.get(severity, "확인 필요")


# 혐오 표현 세부 카테고리 (이미지 기준)
HATE_SPEECH_SUBCATEGORIES = {
    "성_혐오": ["여자는", "남자는", "성차별", "성 고정관념"],
    "연령_차별": ["늙은", "젊은 놈", "아저씨", "아줌마"],
    "인종_지역_혐오": ["지역드립", "전라도", "경상도", "서울 촌놈"],
    "장애인_혐오": ["장애인", "병신", "정신병"],
    "종교_혐오": ["종교", "신앙", "믿음"],
    "정치_혐오": ["정당", "정치인", "좌파", "우파"],
    "직업_혐오": ["직업", "직종"]
}



