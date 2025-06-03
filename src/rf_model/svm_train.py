import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score

from rf_train import load_metadata, extract_all_features

def load_features(features_path, df_meta, fs=1e6):
    if os.path.exists(features_path):
        print(f"Loading precomputed features from {features_path}")
        df_feats = pd.read_csv(features_path)
    else:
        print("Extracting features for all segments...")
        df_feats = extract_all_features(df_meta, fs)
        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        df_feats.to_csv(features_path, index=False)
        print(f"Saved features to {features_path}")
    return df_feats

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    features_path = os.path.join(project_root, 'data', 'rf_features.csv')
    model_dir = os.path.join(project_root, 'models', 'rf')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'svm_model.pkl')

    df_meta = load_metadata()
    df_feats = load_features(features_path, df_meta)

    label_mapping = {'no_drone': 0, 'drone': 1}
    if df_feats['label'].dtype == object:
        df_feats['label'] = df_feats['label'].map(label_mapping)

    X = df_feats.drop('label', axis=1)
    y = df_feats['label']

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)
    print(f"Dataset sizes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('svc', SVC(kernel='linear', probability=True, random_state=42))
    ])
    param_grid = {
        'svc__C': [0.1, 1, 10]
    }

    gs = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', verbose=2, n_jobs=-1)
    print("Starting GridSearchCV for linear SVM...")
    gs.fit(X_train, y_train)

    print("Best parameters:", gs.best_params_)
    y_val_pred = gs.predict(X_val)
    y_val_proba = gs.predict_proba(X_val)[:, 1]
    print("Validation classification report:")
    print(classification_report(y_val, y_val_pred))
    print("Validation ROC AUC:", roc_auc_score(y_val, y_val_proba))

    joblib.dump(gs.best_estimator_, model_path)
    print(f"Saved SVM model to {model_path}")

if __name__ == '__main__':
    main()
