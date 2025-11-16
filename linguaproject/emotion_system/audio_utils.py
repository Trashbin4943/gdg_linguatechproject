from pydub import AudioSegment
import os

def convert_to_wav(audio_path):
    """
    m4a 또는 mp3 파일을 wav로 변환합니다.
    """
    ext = os.path.splitext(audio_path)[1].lower()  # 확장자 추출
    if ext not in [".m4a", ".mp3"]:
        raise ValueError("지원하지 않는 포맷입니다. m4a 또는 mp3만 가능합니다.")

    wav_path = audio_path.replace(ext, ".wav")
    audio = AudioSegment.from_file(audio_path, format=ext.replace(".", ""))
    audio.export(wav_path, format="wav")
    return wav_path
