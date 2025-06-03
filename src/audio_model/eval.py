import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, log_loss,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
    precision_score, recall_score
)
from sklearn.model_selection import train_test_split

from features import extract_audio_features
from AudioDataLoader import create_data_generators


def load_test_set(metadata_path, test_size=0.2, random_state=42):
    df = pd.read_csv(metadata_path)
    df['label'] = df['label'].map({'no_drone': 0, 'drone': 1})
    _, df_test = train_test_split(
        df, test_size=test_size,
        stratify=df['label'], random_state=random_state
    )
    y_test = df_test['label'].values
    return df_test, y_test


def evaluate_classical(df_test, y_test, model_path, output_dir):
    # extract features
    X_test = np.array([
        list(extract_audio_features(row['out_path']).values())
        for _, row in df_test.iterrows()
    ])
    feature_names = list(extract_audio_features(df_test.iloc[0]['out_path']).keys())

    model = joblib.load(model_path)
    name = os.path.splitext(os.path.basename(model_path))[0]

    # predict
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-y_proba))

    # print metrics
    print(f"\n=== {name.upper()} ===")
    print(classification_report(y_test, y_pred, target_names=['no_drone', 'drone']))
    auc = roc_auc_score(y_test, y_proba)
    ll  = log_loss(y_test, y_proba)
    ap  = average_precision_score(y_test, y_proba)
    acc = np.mean(y_pred == y_test)
    print(f"AUC: {auc:.3f}, AP: {ap:.3f}, Accuracy: {acc:.3f}, LogLoss: {ll:.3f}")

    # feature importance if available
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        idxs = np.argsort(importances)[::-1]
        plt.figure(figsize=(10,6))
        plt.bar(np.array(feature_names)[idxs], importances[idxs])
        plt.xticks(rotation=90)
        plt.title(f'{name} Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_feature_importance.png'))
        plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'{name} ROC')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, f'{name}_roc.png'))
    plt.close()

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.step(recall, precision, where='post', label=f'AP={ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{name} Precision-Recall')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(output_dir, f'{name}_pr.png'))
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['no_drone','drone'])
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    plt.title(f'{name} Confusion Matrix')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{name}_cm.png'))
    plt.close(fig)

    return {'name': name, 'fpr': fpr, 'tpr': tpr, 'auc': auc}


def evaluate_cnn(df_test, y_test, model_path, output_dir):
    name = os.path.splitext(os.path.basename(model_path))[0]
    _, _, test_gen = create_data_generators(
        metadata_path=metadata_path,
        shuffle=False, augment=False
    )
    model = tf.keras.models.load_model(model_path)
    # predict
    y_proba = model.predict(test_gen).ravel()
    y_pred  = (y_proba >= 0.5).astype(int)

    # print metrics
    print(f"\n=== {name.upper()} ===")
    print(classification_report(y_test, y_pred, target_names=['no_drone','drone']))
    auc = roc_auc_score(y_test, y_proba)
    ll  = log_loss(y_test, y_proba)
    ap  = average_precision_score(y_test, y_proba)
    acc = np.mean(y_pred == y_test)
    print(f"AUC: {auc:.3f}, AP: {ap:.3f}, Accuracy: {acc:.3f}, LogLoss: {ll:.3f}")

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'{name} ROC')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, f'{name}_roc.png'))
    plt.close()

    return {'name': name, 'fpr': fpr, 'tpr': tpr, 'auc': auc}


if __name__ == '__main__':
    metadata_path = os.path.join('metadata','metadata_audio.csv')
    output_dir   = os.path.join('models','audio','eval_figures')
    os.makedirs(output_dir, exist_ok=True)

    df_test, y_test = load_test_set(metadata_path)
    results = []

    # find model files
    audio_models = [f for f in os.listdir(os.path.join('models','audio'))
                    if f.endswith(('.pkl','.h5','.keras'))]

    for fname in audio_models:
        model_path = os.path.join('models','audio',fname)
        model_type = os.path.splitext(fname)[1]
        # create subdir
        subdir = os.path.join(output_dir, os.path.splitext(fname)[0])
        os.makedirs(subdir, exist_ok=True)
        if fname.endswith(('.pkl')):
            res = evaluate_classical(df_test, y_test, model_path, subdir)
        else:
            res = evaluate_cnn(df_test, y_test, model_path, subdir)
        results.append(res)

    # compare ROC curves
    if results:
        plt.figure(figsize=(8,6))
        for r in results:
            plt.plot(r['fpr'], r['tpr'], label=f"{r['name']} (AUC={r['auc']:.3f})")
        plt.plot([0,1],[0,1],'--', label='Random')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Comparison')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,'roc_comparison.png'))
        plt.close()
    else:
        print("No audio models found for evaluation.")
