import os
import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from rf_train import load_metadata, load_rf_csv


def compute_spectrogram_array(power, fs=1e6, nperseg=256, noverlap=None, size=(128, 128)):
    """
    Compute a fixed-size spectrogram array for CNN input.
    Crops/pads to `size` and normalizes to [0,1].
    """
    freqs, times, Sxx = spectrogram(power, fs=fs, nperseg=nperseg, noverlap=noverlap)
    # Crop frequency dimension
    Sxx = Sxx[:size[0], :]
    # Pad or truncate time dimension
    if Sxx.shape[1] >= size[1]:
        Sxx = Sxx[:, :size[1]]
    else:
        pad_width = size[1] - Sxx.shape[1]
        Sxx = np.pad(Sxx, ((0, 0), (0, pad_width)), mode='constant')
    # Normalize to [0,1]
    max_val = Sxx.max()
    if max_val > 0:
        Sxx = Sxx / max_val
    return Sxx


def load_spectrograms(df_meta, fs=1e6):
    """
    Precompute or load cached spectrogram arrays and labels.
    Saves to data/cnn_spectrograms/*.npy for reuse.
    """
    features_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cnn_spectrograms')
    )
    os.makedirs(features_dir, exist_ok=True)
    X_path = os.path.join(features_dir, 'X.npy')
    y_path = os.path.join(features_dir, 'y.npy')

    if os.path.exists(X_path) and os.path.exists(y_path):
        print(f"Loading precomputed spectrograms from {features_dir}")
        X = np.load(X_path)
        y = np.load(y_path)
    else:
        X_list, y_list = [], []
        for _, row in df_meta.iterrows():
            path = row['path']
            label = row['label']
            _, power_arr = load_rf_csv(path)
            if power_arr.size == 0:
                continue
            spec = compute_spectrogram_array(power_arr, fs)
            # Add channel dimension
            X_list.append(spec[..., np.newaxis])
            y_list.append(1 if label == 'drone' else 0)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        np.save(X_path, X)
        np.save(y_path, y)
        print(f"Saved spectrograms to {features_dir}")

    return X, y


def build_model(input_shape):
    """
    Build a simple CNN for binary classification.
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
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
    # Paths and parameters
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '.', '.')
    )
    model_dir = os.path.join(project_root, 'models', 'rf')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'cnn_model.keras')

    # Load metadata and spectrogram data
    df_meta = load_metadata()
    X, y = load_spectrograms(df_meta)

    # Train/Val/Test split (60/20/20 stratified)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
    )
    print(f"Dataset sizes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Build and train the CNN
    model = build_model(X_train.shape[1:])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32
    )

    # Evaluate on test set
    loss, acc = model.evaluate(X_test, y_test)
    print(f"\n=== Test-set Evaluation ===\nLoss: {loss:.4f}  Accuracy: {acc:.4f}")

    # Save the trained model
    model.save(model_path)
    print(f"Saved CNN model to {model_path}")


if __name__ == '__main__':
    main()
