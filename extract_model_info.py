import os
import joblib
import numpy as np
from tensorflow import keras

def get_file_size_mb(path):
    try:
        size_bytes = os.path.getsize(path)
        return size_bytes / (1024**2)
    except OSError:
        return None

def count_sklearn_params(model):
    # Try to count coefficients and intercepts if available
    total = 0
    if hasattr(model, "coef_"):
        coef = model.coef_
        total += np.prod(coef.shape)
    if hasattr(model, "intercept_"):
        intercept = model.intercept_
        # intercept can be scalar or array
        try:
            total += np.prod(np.array(intercept).shape)
        except Exception:
            total += 1
    # If no coef_/intercept_, try feature_importances_
    if total == 0 and hasattr(model, "feature_importances_"):
        feat = model.feature_importances_
        total += np.prod(feat.shape)
    return int(total) if total > 0 else None

def inspect_models(base_dirs):
    results = []
    for base in base_dirs:
        for root, _, files in os.walk(base):
            for fname in files:
                name, ext = os.path.splitext(fname)
                ext = ext.lower()
                if ext in [".pkl", ".h5", ".keras"]:
                    path = os.path.join(root, fname)
                    size_mb = get_file_size_mb(path)
                    param_count = None
                    if ext in [".h5", ".keras"]:
                        try:
                            model = keras.models.load_model(path)
                            param_count = model.count_params()
                        except Exception as e:
                            param_count = None
                    elif ext == ".pkl":
                        try:
                            model = joblib.load(path)
                            param_count = count_sklearn_params(model)
                        except Exception:
                            param_count = None
                    results.append({
                        "path": path,
                        "size_mb": size_mb,
                        "param_count": param_count
                    })
    return results

def main():
    base_dirs = ["models/audio", "models/fusion", "models/rf"]
    info = inspect_models(base_dirs)
    print(f"{'Path':<60} {'Size (MB)':>10} {'# Params':>10}")
    print("-" * 85)
    for item in info:
        path = item['path']
        size = item['size_mb']
        params = item['param_count']
        size_str = f"{size:.2f}" if size is not None else "N/A"
        params_str = str(params) if params is not None else "N/A"
        print(f"{path:<60} {size_str:>10} {params_str:>10}")

if __name__ == "__main__":
    main()
