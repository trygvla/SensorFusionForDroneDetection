import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras import layers, models

from features import extract_audio_features
from AudioDataLoader import create_data_generators


def build_cnn(input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    return model


def train_classical_models(metadata_path, models_dir):
    df = pd.read_csv(metadata_path)
    df['label'] = df['label'].map({'no_drone': 0, 'drone': 1})

    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        feats = extract_audio_features(row['out_path'])
        X.append(list(feats.values()))
        y.append(row['label'])
    X = np.array(X)
    y = np.array(y)
    feature_names = list(feats.keys())

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipelines = {
        'lr': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=0.95)),
                ('clf', LogisticRegression(max_iter=1000, random_state=42))
            ]),
            {
                'clf__C': [0.01, 0.1, 1, 10],
                'clf__penalty': ['l2']
            }
        ),
        'rf': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=0.95)),
                ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
            ]),
            {
                'pca__n_components': [0.90, 0.95, 0.99],
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [None, 10, 20],
                'clf__min_samples_split': [2, 5]
            }
        ),
        'svm': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=0.95)),
                ('clf', SVC(probability=True, kernel='rbf', random_state=42))
            ]),
            {
                'clf__C': [0.1, 1, 10],
                'clf__gamma': ['scale', 'auto']
            }
        ),
        'xgb': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=0.95)),
                ('clf', xgb.XGBClassifier(tree_method='hist', use_label_encoder=False, eval_metric='logloss', random_state=42))
            ]),
            {
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [3, 6, 10],
                'clf__learning_rate': [0.01, 0.1, 0.2],
                'clf__subsample': [0.6, 0.8, 1.0]
            }
        )
    }

    for name, (pipeline, params) in pipelines.items():
        print(f"\nTraining {name} model...")
        gs = GridSearchCV(
            pipeline, params, cv=5, scoring='roc_auc', verbose=2, n_jobs=-1
        )
        gs.fit(X_train_val, y_train_val)
        print(f"Best {name} params: {gs.best_params_}")
        print(f"Best {name} CV AUC: {gs.best_score_:.4f}")
        out_path = os.path.join(models_dir, f"best_{name}_audio_model.pkl")
        joblib.dump(gs.best_estimator_, out_path)


def train_cnn_model(metadata_path, models_dir, batch_size=32, epochs=50):
    train_gen, val_gen, _ = create_data_generators(
        metadata_path=metadata_path,
        batch_size=batch_size,
        shuffle=True,
        augment=False
    )
    input_shape = train_gen[0][0].shape[1:]
    model = build_cnn(input_shape)

    ckpt_path = os.path.join(models_dir, 'cnn_audio_model.h5')
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', patience=5, mode='max', restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor='val_auc', mode='max', save_best_only=True
        )
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )
    print(f"CNN training complete. Model saved to {ckpt_path}")


if __name__ == '__main__':
    metadata_csv = os.path.join('metadata', 'metadata_audio.csv')
    models_output = os.path.join('models', 'audio')
    os.makedirs(models_output, exist_ok=True)

    train_classical_models(metadata_csv, models_output)
    train_cnn_model(metadata_csv, models_output)
