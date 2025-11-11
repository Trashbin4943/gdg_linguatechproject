import whisper

'''
Whisper model을 활용하여 .wav 확장자 형태의 음성을 텍스트로 변환하는 함수입니다.
'''

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]