import os
import glob
import csv
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# === Konfigurasjon ===
RF_DIR          = "data/rf"
OUT_BASE        = "segments"
METADATA_RF     = "metadata/metadata_rf.csv"
SEGMENT_DUR     = 1.0             # varighet i sekunder per segment
MAX_SEGMENTS    = 4000            # Sett til None for å kjøre alle
SPLIT_RATIOS    = {"train": 0.6, "val": 0.2, "test": 0.2}
RANDOM_STATE    = 42
AUGMENT_METHODS = ["noise"]
NOISE_LEVEL     = 0.01
AUGMENT_EVERY_N = 5               # augment hvert N-te segment

# Kolonnenavn for headerless hackrf_sweep CSV
CSV_COLS = [
    'date', 'time', 'hz_low', 'hz_high', 'hz_bin_width',
    'num_samples', 'dB1', 'dB2', 'dB3', 'dB4', 'dB5'
]

# === Hjelpefunksjoner ===
def find_rf_files(rf_root):
    """
    Returnerer en dict label -> liste av CSV-filer, tilfeldig shufflet.
    """
    file_dict = {}
    for label in os.listdir(rf_root):
        path = os.path.join(rf_root, label)
        if not os.path.isdir(path):
            continue
        files = glob.glob(os.path.join(path, "*.csv"))
        random.seed(RANDOM_STATE)
        random.shuffle(files)
        file_dict[label] = files
    return file_dict


def count_possible_segments(df, seg_dur):
    """
    Beregn antall segmenter basert på tidsstempler.
    """
    total_sec = (df['datetime'].max() - df['datetime'].min()).total_seconds()
    return int(total_sec // seg_dur)


def augment_rf(df_segment, method):
    """
    Enkel støyaugmentering på amplitude (gjennomsnitt av dB-kolonner).
    """
    df_aug = df_segment.copy()
    noise = np.random.normal(0, NOISE_LEVEL, size=len(df_aug))
    df_aug['amplitude'] = df_aug[['dB1','dB2','dB3','dB4','dB5']].mean(axis=1) + noise
    return df_aug


def split_rf_file(label, csv_path, subset, seg_dur, out_base,
                  writer, start_seg_id, remaining_budget,
                  augment=False):
    """
    Segmenterer en RF-CSV i sekvensielle segmenter basert på seg_dur.
    Leser headerless CSV, kombinerer dato og tid til datetime, beregner amplitude.
    """
    # Les CSV uten header
    df = pd.read_csv(csv_path, header=None, names=CSV_COLS)
    # Kombiner date + time til datetime
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    df = df.dropna(subset=['datetime'])

    df.sort_values('datetime', inplace=True)
    df['sec_offset'] = (df['datetime'] - df['datetime'].min()).dt.total_seconds()
    df['amplitude'] = df[['dB1','dB2','dB3','dB4','dB5']].mean(axis=1)

    n_segs = count_possible_segments(df, seg_dur)
    if remaining_budget <= 0 or n_segs <= 0:
        return start_seg_id, 0, 0

    to_make = min(n_segs, remaining_budget)
    seg_id = start_seg_id
    total_written = 0
    basename = os.path.splitext(os.path.basename(csv_path))[0]

    for i in range(to_make):
        start_sec = i * seg_dur
        end_sec   = start_sec + seg_dur
        chunk = df[(df['sec_offset'] >= start_sec) & (df['sec_offset'] < end_sec)]
        if chunk.empty:
            continue

        seg_id += 1
        seg_name = f"seg_{seg_id:05d}"
        out_dir = os.path.join(out_base, seg_name, "rf")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"{basename}.csv")
        chunk.to_csv(out_csv, index=False)

        writer.writerow({
            "segment_id": seg_name,
            "modalitet": "rf",
            "label": label,
            "subset": subset,
            "orig_file": csv_path,
            "start_sec": start_sec,
            "duration": seg_dur,
            "out_path": out_csv
        })
        total_written += 1

        if augment and subset == "train" and (i % AUGMENT_EVERY_N == 0):
            for method in AUGMENT_METHODS:
                try:
                    df_aug = augment_rf(chunk, method)
                    seg_id += 1
                    seg_name_aug = f"seg_{seg_id:05d}_{method}"
                    out_dir_aug = os.path.join(out_base, seg_name_aug, "rf")
                    os.makedirs(out_dir_aug, exist_ok=True)
                    out_csv_aug = os.path.join(out_dir_aug, f"{basename}_{method}.csv")
                    df_aug.to_csv(out_csv_aug, index=False)

                    writer.writerow({
                        "segment_id": seg_name_aug,
                        "modalitet": "rf",
                        "label": label,
                        "subset": subset,
                        "orig_file": csv_path,
                        "start_sec": start_sec,
                        "duration": seg_dur,
                        "out_path": out_csv_aug
                    })
                    total_written += 1
                except Exception as e:
                    print(f"Augment error {method} on {csv_path}: {e}")

    return seg_id, to_make, total_written


if __name__ == "__main__":
    os.makedirs(os.path.dirname(METADATA_RF), exist_ok=True)
    with open(METADATA_RF, "w", newline="") as f:
        fieldnames = [
            "segment_id","modalitet","label",
            "subset","orig_file","start_sec",
            "duration","out_path"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        seg_counter = 0
        total_segments = 0
        files_by_label = find_rf_files(RF_DIR)

        for label, files in files_by_label.items():
            train_files, temp = train_test_split(
                files, test_size=1-SPLIT_RATIOS['train'], random_state=RANDOM_STATE)
            val_files, test_files = train_test_split(
                temp,
                test_size=SPLIT_RATIOS['test']/(SPLIT_RATIOS['val']+SPLIT_RATIOS['test']),
                random_state=RANDOM_STATE
            )

            for subset_name, subset_files in zip(
                ["train","val","test"],
                [train_files,val_files,test_files]
            ):
                if MAX_SEGMENTS is not None:
                    max_per_label = MAX_SEGMENTS // 2
                    seg_budget = int(max_per_label * SPLIT_RATIOS[subset_name])
                else:
                    seg_budget = float('inf')

                written = 0
                for csv_fp in tqdm(subset_files, desc=f"{label}-{subset_name}"):
                    if written >= seg_budget:
                        break
                    remaining = seg_budget - written
                    try:
                        seg_counter, made, written_here = split_rf_file(
                            label, csv_fp, subset_name,
                            SEGMENT_DUR, OUT_BASE,
                            writer, seg_counter,
                            remaining, augment=True
                        )
                        written += written_here
                        total_segments += written_here
                    except Exception as e:
                        print(f"Error processing {csv_fp}: {e}")

                print(f"{label} ({subset_name}): Skrev {written} segmenter")

        print(f"\nFerdig! Genererte totalt {total_segments} segmenter og lagret til '{METADATA_RF}'")
