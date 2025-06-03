import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# --- Konfig ---
data_path   = os.path.join('metadata', 'fusion_dataset.csv')
out_model   = os.path.join('models', 'fusion', 'best_fusion_model.pkl')
test_size   = 0.2
random_seed = 42

# --- Les fusion-datasettet ---
df = pd.read_csv(data_path)
# Konverter labels fra strenger til binære (0=no_drone, 1=drone)
label_map = {'no_drone': 0, 'drone': 1}
df['label'] = df['label'].map(label_map)

X = df[['rf_proba', 'audio_proba']]
y = df['label']

# --- Split train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=random_seed
)

# --- Tren meta-klassifikator ---
meta_clf = LogisticRegression(
    solver='liblinear',
    class_weight='balanced',
    random_state=random_seed
)
meta_clf.fit(X_train, y_train)

# --- Evaluer på testsettet ---
y_pred_proba = meta_clf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba, pos_label=1)
pr_auc = auc(recall, precision)

print(f"Fusion ROC AUC:    {roc_auc:.4f}")
print(f"Fusion PR AUC:     {pr_auc:.4f}")

# --- Kryss-validering på hele settet ---
cv_scores = cross_val_score(
    meta_clf, X, y,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
print(f"5-fold CV ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# --- Lagre modellen ---
os.makedirs(os.path.dirname(out_model), exist_ok=True)
joblib.dump(meta_clf, out_model)
print(f"Lagret fusion-modell til {out_model}")
