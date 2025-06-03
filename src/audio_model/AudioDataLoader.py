import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence


class AudioDataset(Dataset):
    """
    PyTorch Dataset for loading audio files and converting them to log-mel spectrograms.
    Can apply simple augmentations (noise, time-stretch, pitch-shift) if requested.
    """
    def __init__(self,
                 metadata,
                 sr: int = 22050,
                 n_mels: int = 64,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 duration: float = None,
                 augment: bool = False):
        if isinstance(metadata, str):
            self.metadata = pd.read_csv(metadata)
        else:
            self.metadata = metadata.reset_index(drop=True)
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.augment = augment

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = row['out_path']
        label = row['label']

        y, _ = librosa.load(file_path, sr=self.sr, duration=self.duration)
        if self.augment:
            y = self._augment(y)

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)
        X = torch.tensor(log_mel, dtype=torch.float).unsqueeze(0)
        y_tensor = torch.tensor(label, dtype=torch.float)
        return X, y_tensor

    def _augment(self, y: np.ndarray) -> np.ndarray:
        choice = np.random.choice(['noise', 'stretch', 'pitch', 'none'], p=[0.3, 0.2, 0.2, 0.3])
        if choice == 'noise':
            noise = 0.005 * np.random.randn(len(y))
            y = y + noise
        elif choice == 'stretch':
            rate = np.random.uniform(0.8, 1.2)
            y = librosa.effects.time_stretch(y, rate)
        elif choice == 'pitch':
            n_steps = np.random.randint(-2, 3)
            y = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)
        return y


def get_dataloaders(metadata_csv: str,
                    batch_size: int = 32,
                    test_size: float = 0.2,
                    val_size: float = 0.25,
                    random_state: int = 42,
                    num_workers: int = 4,
                    augment: bool = False,
                    **dataset_kwargs):
    df = pd.read_csv(metadata_csv)
    if 'label' in df.columns and not pd.api.types.is_numeric_dtype(df['label']):
        df['label'] = df['label'].map({'no_drone': 0, 'drone': 1})

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df['label'],
        random_state=random_state
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        stratify=train_val['label'],
        random_state=random_state
    )

    train_ds = AudioDataset(train, augment=augment, **dataset_kwargs)
    val_ds = AudioDataset(val, augment=False, **dataset_kwargs)
    test_ds = AudioDataset(test, augment=False, **dataset_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


class AudioDataGenerator(Sequence):
    """
    Keras Sequence for loading audio files and their labels in batches.
    Converts WAV to log-Mel spectrograms.
    """
    def __init__(
        self,
        metadata: pd.DataFrame,
        batch_size: int = 32,
        sr: int = 22050,
        n_mels: int = 64,
        n_fft: int = 2048,
        hop_length: int = 512,
        augment: bool = False,
        shuffle: bool = True
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.batch_size = batch_size
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.metadata) / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.metadata))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx: int):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.metadata.iloc[batch_idx]
        return self.__data_generation(batch_df)

    def __data_generation(self, batch_df: pd.DataFrame):
        X, y = [], []
        for _, row in batch_df.iterrows():
            file_path = row['out_path']
            label = row['label']
            wav, sr = librosa.load(file_path, sr=self.sr)
            mel_spec = librosa.feature.melspectrogram(
                y=wav,
                sr=sr,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
            log_mel = log_mel[..., np.newaxis]
            X.append(log_mel)
            y.append(label)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def create_data_generators(
    metadata_path: str,
    batch_size: int = 32,
    test_size: float = 0.2,
    val_size: float = 0.25,
    **gen_kwargs
):
    df = pd.read_csv(metadata_path)
    df['label'] = df['label'].map({'no_drone': 0, 'drone': 1})

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df['label'],
        random_state=42
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        stratify=train_val['label'],
        random_state=42
    )
    train_gen = AudioDataGenerator(train, batch_size=batch_size, **gen_kwargs)
    val_gen = AudioDataGenerator(val, batch_size=batch_size, **gen_kwargs)
    test_gen = AudioDataGenerator(test, batch_size=batch_size, **gen_kwargs)
    return train_gen, val_gen, test_gen