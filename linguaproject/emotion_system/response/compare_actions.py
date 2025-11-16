'''
권장 조치 vs 실제 조치 비교
difflib.SequenceMatcher로 유사도를 계산합니다.
'''

from difflib import SequenceMatcher

def compare_actions(user_text, recommended_action, actual_action):
    similarity = SequenceMatcher(None, recommended_action, actual_action).ratio()
    return {
        "user_text": user_text,
        "recommended": recommended_action,
        "actual": actual_action,
        "similarity": round(similarity * 100, 2)
    }