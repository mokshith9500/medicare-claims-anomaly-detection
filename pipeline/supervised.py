# supervised RF baseline. point of this: show what a classifier gets when you
# DO have labels, so the unsupervised vs supervised tradeoff is concrete.

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from .features import build_feature_matrix


def supervised_baseline(feat, test_size=0.30, seed=42):
    X, cols = build_feature_matrix(feat)
    y = (feat["PotentialFraud"] == "Yes").astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=seed, n_jobs=-1).fit(X_tr, y_tr)

    proba = rf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)
    base = y_te.mean()

    # precision@10% (same cutoff as the unsupervised screen)
    k = int(0.10 * len(y_te))
    order = np.argsort(proba)[::-1]
    topk = y_te.values[order][:k]
    prec_at_10 = topk.mean()

    importances = pd.Series(rf.feature_importances_, index=cols
                            ).sort_values(ascending=False)

    return {
        "auc": round(float(auc), 3),
        "precision_at_10pct": round(float(prec_at_10), 3),
        "lift_at_10pct": round(float(prec_at_10 / base), 2),
        "base_rate": round(float(base), 3),
        "top_features": importances.head(6).round(3).to_dict(),
    }


if __name__ == "__main__":
    from data_loader import load_data
    from features import build_provider_features
    ip, op, bene, labels, source = load_data()
    feat = build_provider_features(ip, op, bene, labels)
    r = supervised_baseline(feat)
    print(f"[source: {source}] SUPERVISED Random Forest (out-of-sample)")
    print(f"  AUC: {r['auc']}")
    print(f"  precision@10%: {r['precision_at_10pct']:.0%}   lift@10%: {r['lift_at_10pct']}x")
    print(f"  top features: {list(r['top_features'].keys())}")
