import os
import glob
import csv
import math
import random
import soundfile as sf
import numpy as np
from sklearn.model_selection import train_test_split

# === Konfigurasjon ===
AUDIO_DIR       = "data/audio"
OUT_BASE        = "segments"
SEGMENT_DUR     = 1.0
METADATA_AUDIO  = "metadata/metadata_audio.csv"
MAX_SEGMENTS    = 4000  # Sett til None for å kjøre alle
SPLIT_RATIOS    = {"train": 0.6, "val": 0.2, "test": 0.2}
RANDOM_STATE    = 42
AUGMENT_METHODS = ["noise", "pitch"]  # fjernet stretch


def find_audio_files(audio_root):
    file_dict = {}
    for label in os.listdir(audio_root):
        path = os.path.join(audio_root, label)
        if not os.path.isdir(path):
            continue
        files = glob.glob(os.path.join(path, "*.wav"))
        random.seed(RANDOM_STATE)
        random.shuffle(files)
        file_dict[label] = files
    return file_dict


def count_possible_segments(wav_path, seg_dur):
    try:
        info = sf.info(wav_path)
        total_samples = info.frames
        samples_per_seg = int(seg_dur * info.samplerate)
        return total_samples // samples_per_seg
    except:
        return 0


def augment_audio(data, method, sr):
    import librosa
    if method == "noise":
        noise = 0.005 * np.random.randn(len(data))
        return data + noise
    elif method == "pitch":
        return librosa.effects.pitch_shift(data, sr=sr, n_steps=np.random.randint(-2, 3))
    else:
        return data


def split_audio_file(label, wav_path, subset, seg_dur, out_base, writer, start_seg_id, remaining_budget, augment=False):
    data, sr = sf.read(wav_path)
    if data.ndim == 2:
        data = np.mean(data, axis=1)  # konverter stereo til mono
    total_samples = len(data)
    samples_per_seg = int(seg_dur * sr)
    n_available = math.floor(total_samples / samples_per_seg)

    if remaining_budget <= 0:
        return start_seg_id, 0, 0

    max_segments = min(n_available, remaining_budget)
    if max_segments <= 0:
        return start_seg_id, 0, 0

    basename = os.path.splitext(os.path.basename(wav_path))[0]
    seg_id = start_seg_id
    total_written = 0

    for i in range(max_segments):
        start_sample = i * samples_per_seg
        end_sample = start_sample + samples_per_seg
        seg_id += 1
        seg_name = f"seg_{seg_id:05d}"

        out_dir = os.path.join(out_base, seg_name, "audio")
        os.makedirs(out_dir, exist_ok=True)

        clip = data[start_sample:end_sample]
        out_wav = os.path.join(out_dir, f"{basename}.wav")
        sf.write(out_wav, clip, sr)

        writer.writerow({
            "segment_id": seg_name,
            "modalitet": "audio",
            "label": label,
            "subset": subset,
            "orig_file": wav_path,
            "start_sec": i * seg_dur,
            "duration": seg_dur,
            "out_path": out_wav
        })

        total_written += 1

        if augment and subset == "train" and (i % 5 == 0) and len(clip) >= 2048:
            for method in AUGMENT_METHODS:
                try:
                    aug_clip = augment_audio(clip, method, sr)
                    seg_id += 1
                    seg_name_aug = f"seg_{seg_id:05d}_{method}"
                    out_dir_aug = os.path.join(out_base, seg_name_aug, "audio")
                    os.makedirs(out_dir_aug, exist_ok=True)
                    out_wav_aug = os.path.join(out_dir_aug, f"{basename}_{method}.wav")
                    sf.write(out_wav_aug, aug_clip, sr)

                    writer.writerow({
                        "segment_id": seg_name_aug,
                        "modalitet": "audio",
                        "label": label,
                        "subset": subset,
                        "orig_file": wav_path,
                        "start_sec": i * seg_dur,
                        "duration": seg_dur,
                        "out_path": out_wav_aug
                    })
                    total_written += 1
                except Exception as e:
                    print(f"Augment error on {method}: {e}")

    return seg_id, max_segments, total_written


if __name__ == "__main__":
    with open(METADATA_AUDIO, "w", newline="") as f:
        fieldnames = ["segment_id", "modalitet", "label", "subset", "orig_file", "start_sec", "duration", "out_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        seg_counter = 0
        total_segments = 0

        files_by_label = find_audio_files(AUDIO_DIR)

        # Split filene per label
        for label, files in files_by_label.items():
            train_files, temp_files = train_test_split(files, test_size=(1 - SPLIT_RATIOS["train"]), random_state=RANDOM_STATE)
            val_files, test_files = train_test_split(temp_files, test_size=SPLIT_RATIOS["test"] / (SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"]), random_state=RANDOM_STATE)

            for subset_name, subset_files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
                if MAX_SEGMENTS is not None:
                    max_per_class = MAX_SEGMENTS // 2
                    seg_budget = int(max_per_class * SPLIT_RATIOS[subset_name])
                else:
                    seg_budget = float('inf')

                segs_written = 0
                for wav_fp in subset_files:
                    if segs_written >= seg_budget:
                        break
                    remaining_budget = seg_budget - segs_written
                    seg_counter, made, total_written = split_audio_file(label, wav_fp, subset_name, SEGMENT_DUR, OUT_BASE, writer, seg_counter, remaining_budget, augment=True)
                    segs_written += total_written

                print(f"{label} ({subset_name}): Skrev {segs_written} segmenter fra {len(subset_files)} filer")
                total_segments += segs_written

        print(f"\nFerdig! Genererte totalt {total_segments} segmenter og lagret til '{METADATA_AUDIO}'")
