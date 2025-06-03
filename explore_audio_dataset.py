"""
Eksplorativ analyse av audio-datasettet i en .py-fil.

Denne scriptet laster metadata fra metadata/metadata_audio.csv,
viser klassefordeling, plotter mel-spektrogram for begge klasser,
plott av FFT-magnituder for begge klasser,
samt enkel feature-distribusjon (spectral centroid) med labels som tekst.
For å sikre at vi bruker riktig "features.py" fra prosjektet istedenfor et installert "features"-pakke,
legger vi til src/audio_model-mappen i sys.path.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display

# Legg til prosjektets audio_model-katalog i path for å unngå navnekonflikt med en installert features-pakke
script_dir = os.path.dirname(os.path.abspath(__file__))
audio_model_path = os.path.join(script_dir, 'src', 'audio_model')
if os.path.isdir(audio_model_path) and audio_model_path not in sys.path:
    sys.path.insert(0, audio_model_path)

from features import extract_audio_features

def explore_metadata(df):
    total = len(df)
    print(f"Totalt antall klipp: {total}")
    counts = df['label'].value_counts().rename({0: 'no_drone', 1: 'drone'})
    print("\nKlassefordeling:")
    print(counts)
    plt.figure(figsize=(6,4))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title('Antall klipp per klasse')
    plt.ylabel('Antall')
    plt.xlabel('Klasse')
    plt.tight_layout()
    plt.show()


def plot_mel_spectrogram_samples(df):
    print("\nPlotter mel-spektrogram for drone og no_drone...")
    samples = {label: df[df['label'] == label].sample(1, random_state=42)['out_path'].iloc[0]
               for label in sorted(df['label'].unique())}
    for label, path in samples.items():
        y, sr = librosa.load(path, sr=22050)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        S_db = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(8,4))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
        lbl = 'drone' if label == 1 else 'no_drone'
        plt.title(f'Mel-Spectrogram - {lbl}')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()


def plot_fft_samples(df):
    print("\nPlotter FFT-magnitude for drone og no_drone...")
    samples = {label: df[df['label'] == label].sample(1, random_state=42)['out_path'].iloc[0]
               for label in sorted(df['label'].unique())}
    for label, path in samples.items():
        y, sr = librosa.load(path, sr=22050)
        N = len(y)
        fft_vals = np.fft.rfft(y)
        fft_mag = np.abs(fft_vals)
        freqs = np.fft.rfftfreq(N, d=1/sr)
        plt.figure(figsize=(8,4))
        plt.plot(freqs, fft_mag)
        lbl = 'drone' if label == 1 else 'no_drone'
        plt.title(f'FFT Magnitude - {lbl}')
        plt.xlabel('Frekvens (Hz)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.show()


def feature_distributions(df, n=200):
    print("\nEksporterer audio features for sampling...")
    feats = []
    labels = []
    sample_df = df.sample(n, random_state=42)
    for _, row in sample_df.iterrows():
        f = extract_audio_features(row['out_path'])
        feats.append(f)
        labels.append(row['label'])
    feat_df = pd.DataFrame(feats)
    # Mapp label til tekst
    feat_df['label'] = ['drone' if l == 1 else 'no_drone' for l in labels]
    # Plot spectral centroid distribution per klasse
    plt.figure(figsize=(6,4))
    sns.kdeplot(data=feat_df, x='centroid_mean', hue='label', common_norm=False)
    plt.title('Spectral Centroid Mean per klasse')
    plt.xlabel('Centroid mean')
    plt.tight_layout()
    plt.show()


def main():
    metadata_path = os.path.join(script_dir, 'metadata', 'metadata_audio.csv')
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Fant ikke metadata-fil: {metadata_path}")
    df = pd.read_csv(metadata_path)
    df['label'] = df['label'].map({'no_drone': 0, 'drone': 1})

    explore_metadata(df)
    plot_mel_spectrogram_samples(df)
    plot_fft_samples(df)
    feature_distributions(df)

if __name__ == '__main__':
    main()
