import os
from emotion_system.diarization.speaker_split import diarize_and_transcribe
from emotion_system.emotion.text_emotion import classify_text_emotion
from emotion_system.emotion.audio_emotion import classify_audio_emotion
from emotion_system.features.extract_features import extract_features
from emotion_system.response.generate_response import generate_response
from emotion_system.response.compare_actions import compare_actions
from emotion_system.utils.audio_utils import convert_to_wav
from linguaproject.linguaproject.emotion_system.utils.streaming_input import (
    run_live_emotion_only,
    run_live_emotion_with_diarization,
    run_live_pipeline
)
from logic_classify_system.risk_based_classifier import RiskScoreClassifier
from logic_classify_system.classification_criteria import ConsultationMetadata

HF_TOKEN = os.getenv("HF_TOKEN")


def get_user_choice():
    print("🎧 음성 입력 방식을 선택하세요:")
    print("1. 오디오 파일 업로드")
    print("2. 실시간 마이크 입력")
    input_mode = input("입력 번호: ")

    print("\n🧠 처리 방식을 선택하세요:")
    print("A. 감정 분석만 수행")
    print("B. 감정 분석 + 화자 분리")
    print("C. 감정 분석 + 화자 분리 + Risk Score 평가")
    process_mode = input("입력 문자: ").upper()

    return input_mode, process_mode


def run_emotion_only(audio_path):
    print("\n[감정 분석만 수행]")
    # JSON 저장도 함께 수행
    segments = diarize_and_transcribe(audio_path, HF_TOKEN, save_json=True, json_path="emotion_only.json")

    for seg in segments:
        speaker = seg["speaker"]
        text = seg["text"]
        text_emotion = classify_text_emotion(text)
        features = extract_features(audio_path)
        audio_emotion = classify_audio_emotion(features)
        final_emotion = text_emotion if text_emotion else audio_emotion
        response = generate_response(final_emotion, text)

        print(f"[{speaker}] 발화: {text}")
        print(f"감정: {final_emotion}")
        print("응답:", response)
        print("-" * 50)


def run_emotion_with_diarization(audio_path):
    print("\n[감정 분석 + 화자 분리]")
    segments = diarize_and_transcribe(audio_path, HF_TOKEN, save_json=True, json_path="emotion_diarization.json")

    for seg in segments:
        speaker = seg["speaker"]
        text = seg["text"]
        text_emotion = classify_text_emotion(text)
        features = extract_features(audio_path)
        audio_emotion = classify_audio_emotion(features)
        final_emotion = text_emotion if text_emotion else audio_emotion

        print(f"[{speaker}] 발화: {text}")
        print(f"감정: {final_emotion}")
        print("-" * 50)


def run_full_pipeline(audio_path):
    print("\n[감정 분석 + 화자 분리 + Risk Score 평가]")
    segments = diarize_and_transcribe(audio_path, HF_TOKEN, save_json=True, json_path="full_pipeline.json")
    classifier = RiskScoreClassifier()

    for seg in segments:
        speaker = seg["speaker"]
        text = seg["text"]

        # 욕설 필터링
        profanity_result = classifier.profanity_filter.filter_profanity(text)
        if profanity_result:
            print(f"[{speaker}] 발화: {text}")
            print("욕설 감지 → CRITICAL 처리")
            print("Risk Score:", profanity_result.risk_score, profanity_result.risk_level.name)
            print("권장 조치:", profanity_result.recommendation)
            print("-" * 50)
            continue

        # 감정 분석
        text_emotion = classify_text_emotion(text)
        features = extract_features(audio_path)
        audio_emotion = classify_audio_emotion(features)
        final_emotion = text_emotion if text_emotion else audio_emotion

        # Risk Score 평가
        metadata = ConsultationMetadata(
            consultation_content="고충 상담",
            consultation_result="해결 불가",
            requirement_type="다수 요건",
            consultation_reason="업체"
        )
        risk_result = classifier.classify(text, session_context=segments, metadata=metadata)

        # 상담사 응답 생성
        response = generate_response(final_emotion, text)

        # 권장 조치 vs 실제 조치 비교
        comparison = compare_actions(
            text,
            recommended_action="환불 접수 후 3일 내 처리",
            actual_action="처리 지연 중"
        )

        print(f"[{speaker}] 발화: {text}")
        print(f"감정: {final_emotion}")
        print(f"Risk Score: {risk_result.risk_score} ({risk_result.risk_level.name})")
        print("응답:", response)
        print("권장 조치:", risk_result.recommendation)
        print("조치 비교:", comparison)
        print("-" * 50)


def run_pipeline():
    input_mode, process_mode = get_user_choice()

    if input_mode == "1":
        audio_path = input("\n오디오 파일 경로를 입력하세요: ").strip()
        if audio_path.endswith((".m4a", ".mp3")):
            audio_path = convert_to_wav(audio_path)

        if process_mode == "A":
            run_emotion_only(audio_path)
        elif process_mode == "B":
            run_emotion_with_diarization(audio_path)
        elif process_mode == "C":
            run_full_pipeline(audio_path)
        else:
            print("❌ 잘못된 처리 방식입니다.")

    elif input_mode == "2":
        if process_mode == "A":
            run_live_emotion_only()
        elif process_mode == "B":
            run_live_emotion_with_diarization()
        elif process_mode == "C":
            run_live_pipeline()
        else:
            print("❌ 잘못된 처리 방식입니다.")
    else:
        print("❌ 잘못된 입력 방식입니다.")


if __name__ == "__main__":
    run_pipeline()
