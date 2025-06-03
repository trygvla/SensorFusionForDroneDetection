import numpy as np
import librosa
from scipy.signal import welch, spectrogram, butter, filtfilt
import pywt


def extract_audio_features(file_path,
                           sr: int = 22050,
                           n_fft: int = 2048,
                           hop_length: int = 512,
                           n_mfcc: int = 13,
                           n_mels: int = 64,
                           wavelet: str = 'db4',
                           wp_level: int = 3):
    """
    Extract a comprehensive set of audio features from a WAV file.

    Returns a dict of feature_name -> value.
    """
    # Load audio
    y, sr = librosa.load(file_path, sr=sr)
    features = {}

    # 1. Time-domain
    # RMS
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
    features['rms_mean'] = rms.mean()
    features['rms_std'] = rms.std()
    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
    features['zcr_mean'] = zcr.mean()
    features['zcr_std'] = zcr.std()

    # 2. Frequency-domain (Welch PSD)
    freqs, psd = welch(y, fs=sr, nperseg=n_fft)
    features['psd_mean'] = psd.mean()
    features['psd_std'] = psd.std()
    features['psd_max'] = psd.max()

    # 3. Spectral statistics
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85, n_fft=n_fft, hop_length=hop_length)[0]
    features['centroid_mean'] = centroid.mean()
    features['centroid_std'] = centroid.std()
    features['bandwidth_mean'] = bandwidth.mean()
    features['bandwidth_std'] = bandwidth.std()
    features['rolloff_mean'] = rolloff.mean()
    features['rolloff_std'] = rolloff.std()

    # Spectral entropy
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    ps = S / (np.sum(S, axis=0, keepdims=True) + 1e-12)
    entropy = -np.sum(ps * np.log2(ps + 1e-12), axis=0)
    features['spectral_entropy_mean'] = entropy.mean()
    features['spectral_entropy_std'] = entropy.std()

    # 4. MFCC + deltas
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    for i in range(mfcc.shape[0]):
        features[f'mfcc_{i+1}_mean'] = mfcc[i].mean()
        features[f'mfcc_{i+1}_std'] = mfcc[i].std()
    delta_mfcc = librosa.feature.delta(mfcc)
    for i in range(delta_mfcc.shape[0]):
        features[f'delta_mfcc_{i+1}_mean'] = delta_mfcc[i].mean()
        features[f'delta_mfcc_{i+1}_std'] = delta_mfcc[i].std()

    # 5. Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=n_fft, hop_length=hop_length)
    for i in range(chroma.shape[0]):
        features[f'chroma_{i}_mean'] = chroma[i].mean()
        features[f'chroma_{i}_std'] = chroma[i].std()

    # 6. Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    for i in range(contrast.shape[0]):
        features[f'contrast_{i}_mean'] = contrast[i].mean()
        features[f'contrast_{i}_std'] = contrast[i].std()

    # 7. Spectral flux (onset strength)
    flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    features['spectral_flux_mean'] = flux.mean()
    features['spectral_flux_std'] = flux.std()

    # 8. Wavelet Packet Decomposition energies
    wp = pywt.WaveletPacket(data=y, wavelet=wavelet, maxlevel=wp_level)
    for node in wp.get_level(wp_level, 'natural'):
        data = node.data
        features[f'wp_{node.path}_energy'] = np.sum(np.square(data)) / len(data)

    # 9. Mel-spectrogram stats
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel)
    features['mel_mean'] = log_mel.mean()
    features['mel_std'] = log_mel.std()

    # 10. Waterfall (spectrogram) stats
    f, t, Sxx = spectrogram(y, fs=sr, nperseg=n_fft, noverlap=hop_length)
    features['waterfall_mean'] = Sxx.mean()
    features['waterfall_std'] = Sxx.std()
    features['waterfall_max'] = Sxx.max()

    return features



def extract_features(path, sr=None, n_mels=64, **kwargs):
    """
    Henter enkle log‐mel‐features som 1D-array.
    """
    y, sr = librosa.load(path, sr=sr)
    # Beregn mel‐spectrogram og ta log
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels, **kwargs)
    logS = librosa.power_to_db(S, ref=np.max)
    # Flatten eller ta statistikk
    feats = np.concatenate([
        logS.mean(axis=1),
        logS.std(axis=1),
        np.percentile(logS, [10,50,90], axis=1).flatten()
    ])
    return feats

