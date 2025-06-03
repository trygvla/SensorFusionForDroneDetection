import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from features import compute_psd, compute_spectral_statistics, compute_wpd_features, compute_waterfall_features
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def load_metadata(metadata_csv=None):
    """
    Load RF metadata CSV and return DataFrame with normalized file paths and labels.
    """
    if metadata_csv is None:
        base_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(base_dir, '..', '..'))
        metadata_csv = os.path.join(project_root, 'metadata', 'metadata_rf.csv')

    df = pd.read_csv(metadata_csv)
    path_col = 'out_path' if 'out_path' in df.columns else ('path' if 'path' in df.columns else None)
    if path_col is None:
        raise KeyError("Metadata file must contain 'out_path' or 'path' column.")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    df['path'] = df[path_col].apply(lambda x: os.path.normpath(os.path.join(project_root, x)))
    df['label'] = df['label'].astype(str)
    return df[['path', 'label']]


def load_rf_csv(path):
    """
    Load hackrf_sweep CSV, extract only the dB columns based on num_samples header.
    Returns time_arr and power_arr.
    """
    # Read with header row, skip any leading/trailing spaces
    df = pd.read_csv(path, skipinitialspace=True)
    # Ensure the expected columns exist
    if 'num_samples' not in df.columns:
        return np.array([]), np.array([])
    # Number of dB bins
    try:
        n_bins = int(df.loc[0, 'num_samples'])
    except ValueError:
        return np.array([]), np.array([])
    # Identify columns for dB measurements: immediately after 'num_samples'
    cols = list(df.columns)
    start = cols.index('num_samples') + 1
    end = start + n_bins
    # Slice dB columns
    db_df = df.iloc[:, start:end]
    # Convert to numeric and drop empty rows/cols
    db_df = db_df.apply(pd.to_numeric, errors='coerce')
    db_df = db_df.dropna(how='all', axis=1).dropna(how='all', axis=0)
    if db_df.empty:
        return np.array([]), np.array([])
    power_arr = db_df.values.flatten()
    time_arr = np.arange(len(power_arr))
    return time_arr, power_arr


def extract_features(path, fs):
    """
    Extract a full feature vector from a single RF segment.
    """
    _, power_arr = load_rf_csv(path)
    if power_arr.size == 0:
        return None
    # PSD
    freqs, psd = compute_psd(power_arr, fs)
    feats = {f'psd_bin_{i}': val for i, val in enumerate(psd)}
    # Spectral stats
    feats.update(compute_spectral_statistics(freqs, psd))
    # Wavelet packet
    feats.update(compute_wpd_features(power_arr))
    # Waterfall
    feats.update(compute_waterfall_features(power_arr, fs))
    return pd.Series(feats)


def extract_all_features(df_meta, fs):
    """
    Loop over metadata DataFrame, extract features with progress bar.
    """
    feature_list = []
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Extracting RF features"):
        path = row['path']
        label = row['label']
        try:
            feats = extract_features(path, fs)
            if feats is None:
                print(f"Skipping {path}: no power data.")
                continue
            feats['label'] = label
            feature_list.append(feats)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    return pd.DataFrame(feature_list)


def main():
    parser = argparse.ArgumentParser(description="Train Random Forest on RF features")
    parser.add_argument('--metadata', type=str, default=None, help='Path to RF metadata CSV')
    parser.add_argument('--fs', type=float, default=1e6, help='Sampling rate in Hz')
    parser.add_argument('--drop-psd', action='store_true', help='Drop raw PSD bins from features')
    args = parser.parse_args()

    # Load metadata
    df_meta = load_metadata(args.metadata)
    print(f"Loaded {len(df_meta)} metadata entries.")

    # Extract features
    df_feats = extract_all_features(df_meta, args.fs)
    if df_feats.empty:
        raise RuntimeError("No features extracted. Check data and metadata.")

    # Optionally drop PSD bins
    if args.drop_psd:
        psd_cols = [c for c in df_feats.columns if c.startswith('psd_bin_')]
        if psd_cols:
            df_feats.drop(columns=psd_cols, inplace=True)

    # Save features
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    out_path = os.path.join(project_root, 'data', 'rf_features.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_feats.to_csv(out_path, index=False)
    print(f"Features saved to {out_path}.")

    # Split
    X = df_feats.drop('label', axis=1)
    y = df_feats['label']
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    param_grid = {
        'pca__n_components': [0.90, 0.95, 0.99],
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5]
    }
    gs = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', verbose=2, n_jobs=-1)

    # Train
    print("Starting GridSearchCV...")
    gs.fit(X_train, y_train)
    print(f"Best params: {gs.best_params_}")

    # Validate
    # Validation evaluation
    y_val_pred = gs.predict(X_val)
    proba_all   = gs.predict_proba(X_val)
    # Finn riktig kolonne for 'drone'
    pos_idx     = list(gs.classes_).index('drone')
    y_val_proba = proba_all[:, pos_idx]

    y_val_bin = (y_val == 'drone').astype(int)
    # Konverter string-labels til 0/1 (no_drone=0, drone=1)
    y_val_bin = (y_val == 'drone').astype(int)
    # Nå bruker vi y_val_bin som true targets
    print(f"Validation ROC AUC: {roc_auc_score(y_val_bin, y_val_proba):.4f}")

    # Save model
    model_dir = os.path.join(project_root, 'models', 'rf')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'best_rf_model.pkl')
    joblib.dump(gs.best_estimator_, model_path)
    print(f"Saved model to {model_path}.")

if __name__ == '__main__':
    main()
