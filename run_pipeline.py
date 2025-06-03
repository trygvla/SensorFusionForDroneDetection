"""
Run full pipeline from raw data to fusion evaluation.
1. Split raw RF and audio data into segments
2. Merge metadata
3. Generate fusion dataset
4. Train all unimodal RF models
5. Train audio model
6. Evaluate unimodal models
7. Train fusion (stacking) model
8. Evaluate fusion model
"""
import subprocess
import os
import sys

# Change working directory to project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)


def run(cmd):
    print(f"\n>>> Running: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"ERROR: Command failed (exit code {res.returncode}): {cmd}")
        sys.exit(res.returncode)


def main():
    # 1) Split audio into segments (writes metadata_audio.csv)
    run("python src/data_prep/split_audio.py")

    # 2) Split RF into segments (writes metadata_rf.csv)
    run("python src/data_prep/split_rf.py")

    # 3) Merge RF and audio metadata into metadata/metadata.csv
    run("python src/data_prep/merge_metadata.py")

    # 4) Generate fusion dataset (metadata/fusion_dataset.csv)
    run("python -m src.data_prep.generate_fusion_dataset")

    # 5) Train all RF models
    run("python src/rf_model/rf_train.py")
    run("python src/rf_model/cnn_train.py")
    run("python src/rf_model/dnn_train.py")
    run("python src/rf_model/rnn_train.py")
    run("python src/rf_model/svm_train.py")

    # 6) Train audio model
    run("python src/audio_model/train.py")

    # 7) Evaluate unimodal models
    run("python src/rf_model/eval.py")
    run("python src/audio_model/eval.py")

    # 8) Train fusion (stacking)
    run("python src/fusion/stacking.py")

    # 9) Evaluate fusion
    run("python src/fusion/eval_fusion.py")

    print("\nPipeline complete! All models trained and evaluated.")


if __name__ == '__main__':
    main()
