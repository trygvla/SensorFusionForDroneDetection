import os
import pandas as pd
import numpy as np
import joblib
import librosa
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# --- Konfig ---
md_path     = os.path.join('metadata', 'metadata.csv')
out_path    = os.path.join('metadata', 'fusion_dataset.csv')
n_splits    = 5
random_seed = 42

# --- Les metadata og vis kolonner ---
df_meta = pd.read_csv(md_path)
print("Metadata-kolonner:", df_meta.columns.tolist())

# --- Importer RF-ekstraktor og audio-ekstraktor ---
from src.rf_model.features import extract_features_from_file as rf_extract

# --- Last inn modeller ---
# Her må den beste RF og AUdio modellen velges manuelt!

rf_model    = joblib.load('models/rf/best_rf_model.pkl')
# Velg én av de trenede audio-modellene:
# e.g., best logistic regression, random forest, SVM eller XGBoost
audio_model = joblib.load('models/audio/best_rf_audio_model.pkl')
# audio_model = joblib.load('models/audio/best_rf_audio_model.pkl')
# audio_model = joblib.load('models/audio/best_svm_audio_model.pkl')
# audio_model = joblib.load('models/audio/best_xgb_audio_model.pkl')

# --- Initialiser lister for features og labels ---
rf_feats, audio_feats, labels, seg_ids = [], [], [], []

# --- Ekstraher features per segment ---
for _, row in df_meta.iterrows():
    rf_path    = row['out_path_rf']
    audio_path = row['out_path_audio']
    label      = row['label_rf']
    seg_id     = row['segment_id']

    # RF-features (Series -> numpy array)
    series_rf = rf_extract(rf_path, fs=1e6)
    rf_feats.append(series_rf.values)

    # Audio-features: log-mel-statistikk
    y, sr = librosa.load(audio_path, sr=None)
    S     = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=2048, hop_length=512)
    logS  = librosa.power_to_db(S)
    feat_audio = np.concatenate([logS.mean(axis=1), logS.std(axis=1)])
    audio_feats.append(feat_audio)

    labels.append(label)
    seg_ids.append(seg_id)

# --- Konverter til DataFrame/Series ---
X_rf    = pd.DataFrame(rf_feats)
X_audio = pd.DataFrame(audio_feats)
y       = pd.Series(labels)

# --- Out-of-fold prediksjoner ---
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
rf_oof    = cross_val_predict(rf_model,    X_rf,    y, cv=kf, method='predict_proba')[:, 1]
audio_oof = cross_val_predict(audio_model, X_audio, y, cv=kf, method='predict_proba')[:, 1]

# --- Lag fusion-dataset og skriv til CSV ---
df_fusion = pd.DataFrame({
    'segment_id':  seg_ids,
    'rf_proba':    rf_oof,
    'audio_proba': audio_oof,
    'label':       y
})
df_fusion.to_csv(out_path, index=False)
print(f"Fusion dataset skrevet til {out_path}")
