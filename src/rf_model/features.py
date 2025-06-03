import numpy as np
import pandas as pd
from scipy.signal import welch, spectrogram
import pywt
from sklearn.decomposition import PCA



def load_rf_csv(path, *args, **kwargs):
    """
    Leser hackrf_sweep CSV uten header, fjerner initial whitespace
    og returnerer:
      - time_arr: enkel indeks (0,1,2,…)
      - power_arr: flattet 1D-array med alle dB-målinger
    Forventede kolonner per rad:
      0: date (str)
      1: time (str)
      2: hz_low (float)
      3: hz_high (float)
      4: hz_bin_width (float)
      5: num_samples (int)
      6..N: dB-målinger (float)
    """
    import pandas as pd
    import numpy as np

    # Les alle kolonner som data (ingen header), fjern eventuelle mellomrom
    df = pd.read_csv(path, header=None, skipinitialspace=True)

    if df.shape[1] <= 6:
        return np.array([]), np.array([])

    # Ta ut alle dB-kolonner
    db_df = df.iloc[:, 6:]

    # Konverter til float og dropp nan-kolonner/rader
    db_df = db_df.apply(pd.to_numeric, errors='coerce').dropna(how='all', axis=1).dropna(how='all', axis=0)
    if db_df.empty:
        return np.array([]), np.array([])

    # Flatt ut til 1D-array
    power_arr = db_df.values.flatten()
    time_arr  = np.arange(len(power_arr))

    return time_arr, power_arr


def compute_psd(power, fs, nperseg=256, noverlap=None):
    """
    Compute Power Spectral Density using Welch's method.
    Returns frequencies and PSD values.
    """
    freqs, psd = welch(power, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return freqs, psd


def compute_spectral_statistics(freqs, psd):
    """
    Compute spectral statistics: centroid, bandwidth, roll-off, entropy.
    Returns a dict of features.
    """
    psd_norm = psd / np.sum(psd)
    centroid = np.sum(freqs * psd_norm)
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd_norm))
    cumulative = np.cumsum(psd_norm)
    rolloff_freq = freqs[np.searchsorted(cumulative, 0.85)]
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    return {
        'centroid': centroid,
        'bandwidth': bandwidth,
        'rolloff': rolloff_freq,
        'entropy': entropy
    }


def compute_wpd_features(power, wavelet='db4', maxlevel=3, feature_funcs=None):
    """
    Compute Wavelet Packet Decomposition features.
    Returns a dict of energy per node or custom features.
    """
    wp = pywt.WaveletPacket(data=power, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    nodes = wp.get_level(maxlevel, order='freq')
    features = {}
    for i, node in enumerate(nodes):
        data = node.data
        energy = np.sum(data ** 2)
        features[f'wpd_energy_l{maxlevel}_n{i}'] = energy
        if feature_funcs:
            for name, func in feature_funcs.items():
                features[f'{name}_l{maxlevel}_n{i}'] = func(data)
    return features


def compute_waterfall_features(power, fs, nperseg=256, noverlap=None, agg_funcs=None):
    """
    Compute waterfall (time-frequency) features.
    Returns aggregated statistics over the spectrogram matrix.
    """
    freqs, times, Sxx = spectrogram(power, fs=fs, nperseg=nperseg, noverlap=noverlap)
    features = {}
    if agg_funcs is None:
        agg_funcs = {
            'mean': np.mean,
            'std': np.std,
            'max': np.max
        }
    for name, func in agg_funcs.items():
        features[f'waterfall_{name}'] = func(Sxx)
    return features


def compute_pca_features(feature_matrix, n_components=5):
    """
    Apply PCA to feature matrix (samples x features).
    Returns the PCA-transformed matrix (samples x n_components) and explained variance ratio.
    """
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(feature_matrix)
    evr = pca.explained_variance_ratio_
    df_pca = pd.DataFrame(transformed, columns=[f'pca_{i+1}' for i in range(n_components)])
    df_pca['explained_variance_ratio'] = np.sum(evr)
    return df_pca, evr


def extract_features_from_file(path, fs, time_column=None, power_column=None):
    """
    Load a CSV and extract a full feature vector.
    """
    time, power = load_rf_csv(path, time_column, power_column)
    freqs, psd = compute_psd(power, fs)
    features = {}
    # PSD values as features
    for i, val in enumerate(psd):
        features[f'psd_bin_{i}'] = val
    # Spectral statistics
    features.update(compute_spectral_statistics(freqs, psd))
    # WPD
    features.update(compute_wpd_features(power))
    # Waterfall
    features.update(compute_waterfall_features(power, fs))
    return pd.Series(features)

# Example usage:
# df_feats = extract_features_from_file('data/rf/drone/rfdronefile1.csv', fs=1e6)  # fs in Hz


def extract_features(path, fs=1e6):
    """
    Henter alle RF‐features som én 1D-array.
    path: sti til hackrf_sweep‐CSV
    fs: samplerate i Hz (juster om nødvendig)
    """
    feat_series = load_rf_csv(path, fs=fs)
    return feat_series.values  # numpy‐array