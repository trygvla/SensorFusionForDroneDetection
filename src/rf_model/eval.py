import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    log_loss,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split


def load_features():
    """
    Last inn funksjoner inklusive PSD-bin kolonner fra csv.
    """
    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'rf_features.csv')
    )
    df = pd.read_csv(data_path)
    X = df.drop('label', axis=1)
    y = df['label'].map({'no_drone': 0, 'drone': 1})
    return X, y


def split_data(X, y):
    """
    Stratified split: 60% train, 20% val, 20% test.
    """
    _, X_tmp, _, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
    )
    return X_test, y_test


def plot_and_save(fig, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    plt.close(fig)


def evaluate_model(name, model_path, X_test, y_test, results, feature_names, output_dir=None):
    if not os.path.exists(model_path):
        print(f"WARNING: Skipping {name}, modell ikke funnet: {model_path}")
        return

    # Last inn modell
    if model_path.endswith(('.h5', '.keras')):
        model = tf.keras.models.load_model(model_path)
        scaler_path = os.path.join(os.path.dirname(model_path), 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            X_eval = scaler.transform(X_test)
        else:
            print(f"WARNING: Skaleringsfil ikke funnet ({scaler_path}), bruker rå X_test")
            X_eval = X_test.values
        y_proba = model.predict(X_eval).ravel()
        y_pred = (y_proba >= 0.5).astype(int)
    else:
        model = joblib.load(model_path)
        if not hasattr(model, 'predict'):
            print(f"WARNING: Skipping {name}, ingen predict() funnet")
            return
        raw_pred = model.predict(X_test)
        if isinstance(raw_pred.flat[0], str):
            mapping = {'no_drone': 0, 'drone': 1}
            y_pred = np.array([mapping[p] for p in raw_pred])
        else:
            y_pred = raw_pred

        if hasattr(model, 'predict_proba'):
            proba_all = model.predict_proba(X_test)
            classes = getattr(model, 'classes_', None)
            pos_idx = list(classes).index('drone') if classes is not None and 'drone' in classes else 1
            y_proba = proba_all[:, pos_idx]
        elif hasattr(model, 'decision_function'):
            df_scores = model.decision_function(X_test)
            y_proba = 1 / (1 + np.exp(-df_scores))
        else:
            print(f"WARNING: Skipping {name}, ingen predict_proba eller decision_function")
            return

        # Feature importances for tree-based eller lineære koeff
        clf = model.steps[-1][1] if isinstance(model, Pipeline) else model
        if hasattr(clf, 'feature_importances_') or hasattr(clf, 'coef_'):
            if hasattr(clf, 'feature_importances_'):
                importances = clf.feature_importances_
            else:
                importances = np.abs(clf.coef_[0])
            idxs = np.argsort(importances)[::-1]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(np.array(feature_names)[idxs], importances[idxs])
            ax.set_xticklabels(np.array(feature_names)[idxs], rotation=90)
            ax.set_ylabel('Importance')
            ax.set_title(f'{name} Feature Importances')
            plot_and_save(fig, f'{name}_feature_importance.png', output_dir)

    # Beregn metrikker
    report = classification_report(y_test, y_pred, target_names=['no_drone', 'drone'])
    auc = roc_auc_score(y_test, y_proba)
    ll = log_loss(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    precision, recall, pr_thresh = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    results[name] = {
        'report': report,
        'auc': auc,
        'logloss': ll,
        'brier': brier,
        'pr_auc': ap,
        'fpr': roc_curve(y_test, y_proba)[0],
        'tpr': roc_curve(y_test, y_proba)[1],
        'precision': precision,
        'recall': recall
    }

    # Print
    print(f"\n=== {name} Evaluation ===")
    print(report)
    print(f"AUC: {auc:.3f}  LogLoss: {ll:.3f}  Brier: {brier:.3f}  PR AUC: {ap:.3f}")

    # Plot ROC
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(results[name]['fpr'], results[name]['tpr'], label=f'AUC={auc:.3f}')
    ax.plot([0, 1], [0, 1], '--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{name} ROC Curve')
    ax.legend(loc='lower right')
    plot_and_save(fig, f'{name}_roc_curve.png', output_dir)

    # Plot Precision-Recall
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.step(precision, recall, where='post', label=f'AP={ap:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{name} Precision-Recall Curve')
    ax.legend(loc='lower left')
    plot_and_save(fig, f'{name}_pr_curve.png', output_dir)

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['no_drone', 'drone'])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    ax.set_title(f'{name} Confusion Matrix')
    plot_and_save(fig, f'{name}_confusion_matrix.png', output_dir)


def main():
    X, y = load_features()
    feature_names = X.columns.tolist()
    X_test, y_test = split_data(X, y)

    base = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'rf')
    )
    model_files = {
        os.path.splitext(f)[0]: os.path.join(base, f)
        for f in os.listdir(base)
        if f.endswith(('.pkl', '.h5', '.keras'))
    }

    if not model_files:
        print("Ingen modeller funnet i ", base)
        return

    results = {}
    out_fig_dir = os.path.join(base, 'eval_figures')

    for name, path in sorted(model_files.items()):
        evaluate_model(name, path, X_test, y_test, results, feature_names, output_dir=out_fig_dir)

    # Sammendrag
    if results:
        summary = pd.DataFrame([
            {'model': name,
             'AUC': res['auc'],
             'LogLoss': res['logloss'],
             'Brier': res['brier'],
             'PR_AUC': res['pr_auc']}
            for name, res in results.items()
        ])
        summary = summary.sort_values(by='AUC', ascending=False)
        print("\n=== Sammendrag av alle modeller ===")
        print(summary.to_string(index=False))
        summary.to_csv(os.path.join(out_fig_dir, 'summary_metrics.csv'), index=False)

if __name__ == '__main__':
    main()
