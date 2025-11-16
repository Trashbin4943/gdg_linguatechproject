import whisper
from pyannote.audio import Pipeline
import json

'''
Whisper로 Speech-To-Text
Pyannote로 화자분리(diarization) 수행 + JSON 저장 (타임스탬프 포함)
'''

def diarize_and_transcribe(audio_path, hf_token, save_json=False, json_path="segments.json"):
    # 모델 로드
    whisper_model = whisper.load_model("medium")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)

    # 화자 분리 수행
    diarization = pipeline(audio_path)
    segments = []

    # 화자별 구간 추출
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        result = whisper_model.transcribe(audio_path)
        segments.append({
            "speaker": speaker,
            "start": turn.start,   # 시작 시간 (초)
            "end": turn.end,       # 종료 시간 (초)
            "text": result["text"]
        })

    # JSON 저장 옵션
    if save_json:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=4)

    return segments
