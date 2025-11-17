import os
import queue
import sounddevice as sd
import numpy as np
import threading
import tempfile
import wave

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from ..emotion.text_emotion import classify_text_emotion
from ..emotion.audio_emotion import classify_audio_emotion
from ..features.extract_features import extract_features
from ..response.generate_response import generate_response
from logic_classify_system.risk_based_classifier import RiskScoreClassifier, ConsultationMetadata
HF_TOKEN = os.getenv("HF_TOKEN")

# 오디오 큐
audio_queue = queue.Queue()

# 모델 초기화
whisper_model = WhisperModel("medium", device="cuda", compute_type="float16")
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)
classifier = RiskScoreClassifier()


def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())


def save_temp_wav(audio_data, samplerate=16000):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_file.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
    return temp_file.name


def run_live_emotion_only():
    """실시간 감정 분석만 수행"""
    stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=16000)
    stream.start()

    def emotion_only_loop():
        while True:
            if not audio_queue.empty():
                audio_chunk = audio_queue.get()
                audio_data = np.squeeze(audio_chunk)
                segments, _ = whisper_model.transcribe(audio_data, language="ko")
                for segment in segments:
                    text = segment.text
                    temp_wav = save_temp_wav(audio_data)
                    features = extract_features(temp_wav)
                    text_emotion = classify_text_emotion(text)
                    audio_emotion = classify_audio_emotion(features)
                    final_emotion = text_emotion if text_emotion else audio_emotion
                    print(f"발화: {text}")
                    print(f"감정: {final_emotion}")
                    print("-" * 50)

    threading.Thread(target=emotion_only_loop, daemon=True).start()
    print("🎙️ 실시간 감정 분석 시작 (Ctrl+C로 종료)")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("🛑 실시간 입력 종료")
        stream.stop()


def run_live_emotion_with_diarization():
    """실시간 감정 분석 + 화자 분리"""
    stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=16000)
    stream.start()

    def emotion_diarization_loop():
        while True:
            if not audio_queue.empty():
                audio_chunk = audio_queue.get()
                audio_data = np.squeeze(audio_chunk)
                segments, _ = whisper_model.transcribe(audio_data, language="ko")
                diarization = diarization_pipeline(audio_data)
                for segment in segments:
                    text = segment.text
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        temp_wav = save_temp_wav(audio_data)
                        features = extract_features(temp_wav)
                        text_emotion = classify_text_emotion(text)
                        audio_emotion = classify_audio_emotion(features)
                        final_emotion = text_emotion if text_emotion else audio_emotion
                        print(f"[{speaker}] 발화: {text}")
                        print(f"감정: {final_emotion}")
                        print("-" * 50)

    threading.Thread(target=emotion_diarization_loop, daemon=True).start()
    print("🎙️ 실시간 감정 분석 + 화자 분리 시작 (Ctrl+C로 종료)")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("🛑 실시간 입력 종료")
        stream.stop()


def run_live_pipeline():
    """실시간 전체 파이프라인"""
    stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=16000)
    stream.start()

    def full_loop():
        while True:
            if not audio_queue.empty():
                audio_chunk = audio_queue.get()
                audio_data = np.squeeze(audio_chunk)
                segments, _ = whisper_model.transcribe(audio_data, language="ko")
                diarization = diarization_pipeline(audio_data)
                for segment in segments:
                    text = segment.text
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        print(f"[{speaker}] {text}")

                        # 욕설 필터링
                        profanity_result = classifier.profanity_filter.filter_profanity(text)
                        if profanity_result:
                            print("욕설 감지 → CRITICAL 처리")
                            print("Risk Score:", profanity_result.risk_score, profanity_result.risk_level.name)
                            print("권장 조치:", profanity_result.recommendation)
                            print("-" * 50)
                            continue

                        # 감정 분석
                        temp_wav = save_temp_wav(audio_data)
                        features = extract_features(temp_wav)
                        text_emotion = classify_text_emotion(text)
                        audio_emotion = classify_audio_emotion(features)
                        final_emotion = text_emotion if text_emotion else audio_emotion

                        # Risk Score 평가
                        metadata = ConsultationMetadata(
                            consultation_content="실시간 상담",
                            consultation_result="추가 상담 필요",
                            requirement_type="단일 요건",
                            consultation_reason="일반"
                        )
                        risk_result = classifier.classify(text, metadata=metadata)

                        # 응답 생성
                        response = generate_response(final_emotion, text)

                        # 출력
                        print(f"감정: {final_emotion}")
                        print(f"Risk Score: {risk_result.risk_score} ({risk_result.risk_level.name})")
                        print("응답:", response)
                        print("권장 조치:", risk_result.recommendation)
                        print("-" * 50)

    threading.Thread(target=full_loop, daemon=True).start()
    print("🎙️ 실시간 전체 파이프라인 시작 (Ctrl+C로 종료)")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("🛑 실시간 입력 종료")
        stream.stop()
