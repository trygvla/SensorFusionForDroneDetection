import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# TensorFlow/Keras imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Reuse existing RF feature extraction
from rf_train import load_metadata, extract_all_features

def load_features(features_path, df_meta, fs=1e6):
    """
    Load or extract RF features for all segments, same as DNN.
    """
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


def build_model(input_dim):
    """
    Build a simple RNN-based model mirroring the DNN complexity.
    Treat feature vector as sequence of length input_dim with 1 feature per timestep.
    """
    model = Sequential([
        Input(shape=(input_dim, 1)),
        GRU(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        GRU(64),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    # Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    features_path = os.path.join(project_root, 'data', 'rf_features.csv')
    model_dir = os.path.join(project_root, 'models', 'rf')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'rnn_model.keras')
    scaler_path = os.path.join(model_dir, 'rnn_scaler.pkl')

    # Load metadata and features
    df_meta = load_metadata()
    df_feats = load_features(features_path, df_meta)

    # Prepare labels
    if 'label' not in df_feats.columns:
        raise KeyError("Features dataframe must contain 'label' column.")
    label_map = {'no_drone': 0, 'drone': 1}
    if df_feats['label'].dtype == object:
        df_feats['label'] = df_feats['label'].map(label_map)

    # Feature matrix and target vector
    X = df_feats.drop('label', axis=1).values
    y = df_feats['label'].values

    # Train/Val/Test split (60/20/20 stratified)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
    )
    print(f"Dataset sizes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")

    # Reshape for RNN input: (samples, timesteps, features=1)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Compute class weights
    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, cw))
    print(f"Using class weights: {class_weights}")

    # Build and train the RNN model
    model = build_model(X_train.shape[1])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # Evaluate on test set
    y_proba = model.predict(X_test).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    print("\n=== Test-set Evaluation ===")
    print(classification_report(y_test, y_pred, target_names=['no_drone', 'drone']))
    print("Test ROC AUC:", roc_auc_score(y_test, y_proba))

    # Save final model
    model.save(model_path)
    print(f"Saved RNN model to {model_path}")

if __name__ == '__main__':
    main()
