from stt import transcribe_audio
from text_emotion import analyze_text_emotion
from audio_features import extract_features
from audio_models import build_lstm, predict_audio_emotion
from ensemble import combine_emotions
from response_generator import generate_response

'''
전체 메인 함수입니다.
'''

emotion_map = {
'불만': 0, '분노': 1, '불안': 2, '중립': 3,
'감사': 4, '요청': 5, '혼란': 6
}

audio_path = "audio_folder/sample.wav"
text = transcribe_audio(audio_path)
text_emotion = analyze_text_emotion(text)
features = extract_features(audio_path)
audio_model = build_lstm(input_dim=features.shape[0], num_classes=12)
audio_emotion = predict_audio_emotion(audio_model, features)
final_emotion = combine_emotions(text_emotion, audio_emotion)
response = generate_response(final_emotion, text)

print("🗣️ 사용자 발화:", text)
print("🎯 텍스트 감정:", emotion_map[text_emotion])
print("🔊 음향 감정:", emotion_map[audio_emotion])
print("✅ 최종 감정:", emotion_map[final_emotion])
print("💬 상담사 응답:", response)
