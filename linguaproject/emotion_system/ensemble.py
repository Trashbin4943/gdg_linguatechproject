'''
텍스트 추출 감정과 음향 추출 감정을 비교하여 최종 감정을 결정하는 함수입니다.
텍스트 감정은 text_emotion.py, 음향 감정은 audio_models.py에서 정의되어있습니다.
'''

def combine_emotions(text_emotion, audio_emotion):
    if text_emotion == audio_emotion:
        return text_emotion
    else:
        return text_emotion  # 텍스트 기반 우선