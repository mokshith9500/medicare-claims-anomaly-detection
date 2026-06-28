"""
validate.py
-----------
The honesty check. The detector never sees the PotentialFraud label. Here we
ask: how well do the unsupervised flags line up with known fraud?

The headline metric for an unsupervised screen is LIFT: among providers we
flagged, what share are actually fraud, versus the base rate in the whole
population. Lift of 4x means a reviewer working our flagged list finds fraud
four times more often than working the full list blindly. That is the real
operational value, and it is an honest framing because we are not claiming to
"catch all fraud", only to triage review effort far more efficiently.
"""

import numpy as np
import pandas as pd


def validate(res):
    """res: output of detect_anomalies (must contain 'flagged' and 'PotentialFraud')."""
    y = (res["PotentialFraud"] == "Yes")
    flag = res["flagged"]

    base_rate = y.mean()
    n_flag = int(flag.sum())

    tp = int((flag & y).sum())
    fp = int((flag & ~y).sum())
    fn = int((~flag & y).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0   # of flagged, % truly fraud
    recall = tp / (tp + fn) if (tp + fn) else 0.0      # of fraud, % we caught
    lift = precision / base_rate if base_rate else 0.0

    report = {
        "providers": len(res),
        "base_fraud_rate": round(base_rate, 4),
        "flagged": n_flag,
        "flagged_pct": round(n_flag / len(res), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "lift_over_base_rate": round(lift, 2),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }
    return report


def by_anomaly_score(res, k_values=(50, 100, 150)):
    """Precision among the top-k highest-scoring providers (triage view)."""
    ranked = res.sort_values("anomaly_score", ascending=False)
    y = (ranked["PotentialFraud"] == "Yes").values
    out = []
    for k in k_values:
        topk = y[:k]
        out.append({"top_k": k, "precision": round(topk.mean(), 3),
                    "fraud_found": int(topk.sum())})
    return out


def print_report(res):
    rep = validate(res)
    print("=" * 52)
    print(" VALIDATION  (flags vs ground-truth PotentialFraud)")
    print("=" * 52)
    print(f" Providers analyzed........ {rep['providers']}")
    print(f" Base fraud rate........... {rep['base_fraud_rate']:.1%}")
    print(f" Providers flagged......... {rep['flagged']} ({rep['flagged_pct']:.1%})")
    print("-" * 52)
    print(f" Precision (flag is fraud). {rep['precision']:.1%}")
    print(f" Recall (fraud we caught).. {rep['recall']:.1%}")
    print(f" LIFT over base rate....... {rep['lift_over_base_rate']}x")
    print("-" * 52)
    print(f" True positives............ {rep['true_positives']}")
    print(f" False positives........... {rep['false_positives']}")
    print(f" False negatives........... {rep['false_negatives']}")
    print("=" * 52)
    print("\n Triage view (precision among highest-scoring providers):")
    for row in by_anomaly_score(res):
        print(f"   top {row['top_k']:>3}:  {row['precision']:.0%} fraud "
              f"({row['fraud_found']} of {row['top_k']})")
    return rep


if __name__ == "__main__":
    from data_loader import load_data
    from features import build_provider_features
    from anomaly import detect_anomalies

    ip, op, bene, labels, source = load_data()
    feat = build_provider_features(ip, op, bene, labels)
    res = detect_anomalies(feat)
    print(f"\n[data source: {source}]\n")
    print_report(res)
