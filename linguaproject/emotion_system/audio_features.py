import librosa
import numpy as np

'''
음성학 요소를 추출하는 함수입니다.
pitch(피치), energy(에너지), spec_centroid(주파수 분포), 
zcr(신호의 진동성), speech_rate(말 속도), mfcc_1~13(주파수 기반 음색 정보)
'''

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    energy = np.mean(librosa.feature.rms(y=y))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    duration = librosa.get_duration(y=y, sr=sr)
    speech_rate = len(librosa.effects.split(y)) / duration

    features = {
        'pitch': pitch,
        'energy': energy,
        'spec_centroid': spec_centroid,
        'zcr': zcr,
        'speech_rate': speech_rate
    }
    for i, val in enumerate(mfccs_mean):
        features[f'mfcc_{i+1}'] = val
    return np.array(list(features.values()))