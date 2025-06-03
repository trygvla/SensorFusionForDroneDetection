import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    log_loss  # <— lagt til
)

# --- Konfigurasjon ---
data_path   = os.path.join('metadata', 'fusion_dataset.csv')
model_path  = os.path.join('models', 'fusion', 'best_fusion_model.pkl')
output_dir  = os.path.join('src', 'fusion', 'figures')
os.makedirs(output_dir, exist_ok=True)

# --- Last data ---
df = pd.read_csv(data_path)
# Map labels fra str til binær
df['label_bin'] = df['label'].map({'no_drone': 0, 'drone': 1})
X = df[['rf_proba', 'audio_proba']]
y = df['label_bin']

# --- Last modell ---
clf = joblib.load(model_path)

# --- Prediksjoner ---
y_proba = clf.predict_proba(X)[:, 1]
y_pred  = clf.predict(X)

# --- Beregn LogLoss ---
ll = log_loss(y, y_proba)  # <— beregning av logloss

# --- ROC-kurve ---
fpr, tpr, _ = roc_curve(y, y_proba)
roc_auc     = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'Fusion ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Fusion Model ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'fusion_roc_curve.png'))

# --- Precision-Recall-kurve ---
precision, recall, _ = precision_recall_curve(y, y_proba, pos_label=1)
pr_auc             = auc(recall, precision)
plt.figure()
plt.plot(recall, precision, label=f'Fusion PR (AUC = {pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Fusion Model Precision-Recall Curve')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'fusion_pr_curve.png'))

# --- Forvirringsmatrise ---
cm = confusion_matrix(y, y_pred)
labels = ['no_drone', 'drone']
fig, ax = plt.subplots()
cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.set_xticks([0, 1])
ax.set_xticklabels(labels)
ax.set_yticks([0, 1])
ax.set_yticklabels(labels)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j],
                ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Fusion Model Confusion Matrix')
fig.colorbar(cax)
plt.savefig(os.path.join(output_dir, 'fusion_confusion_matrix.png'))

# --- Klassifikasjonsrapport ---
print("--- Fusion Model Metrics ---")
print(f"ROC AUC:            {roc_auc:.4f}")
print(f"PR AUC:             {pr_auc:.4f}")
print(f"LogLoss:            {ll:.4f}")  # <— print logloss
print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=labels))

print(f"Figurer lagret i {output_dir}")
