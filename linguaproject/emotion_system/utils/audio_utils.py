from pydub import AudioSegment
import os

def convert_to_wav(file_path: str) -> str:
    """
    m4a 또는 mp3 파일을 wav로 변환합니다.

    Args:
        file_path: 원본 오디오 파일 경로 (.m4a 또는 .mp3)

    Returns:
        변환된 wav 파일 경로
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in [".m4a", ".mp3"]:
        raise ValueError("지원되지 않는 오디오 형식입니다. m4a 또는 mp3만 가능합니다.")

    wav_path = file_path.replace(ext, ".wav")
    audio = AudioSegment.from_file(file_path, format=ext[1:])  # 'm4a' 또는 'mp3'
    audio.export(wav_path, format="wav")
    return wav_path