# Runs the whole thing and writes outputs/ (scored csv, report.json, map png).
# Headline metrics are out-of-sample.

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

from pipeline.data_loader import load_data
from pipeline.features import build_provider_features, build_feature_matrix
from pipeline.anomaly import detect_with_holdout, detect_full, stability
from pipeline.supervised import supervised_baseline

OUT = os.path.join(os.path.dirname(__file__), "outputs")


def evaluate(res):
    y = (res["PotentialFraud"] == "Yes")
    base = y.mean()
    flag = res["flagged"]
    tp = int((flag & y).sum()); fp = int((flag & ~y).sum()); fn = int((~flag & y).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    ranked = res.sort_values("anomaly_score", ascending=False)
    yr = (ranked["PotentialFraud"] == "Yes").values
    triage = [{"top_k": k, "precision": round(float(yr[:k].mean()), 3),
               "lift": round(float(yr[:k].mean() / base), 2)}
              for k in (25, 50, 100, 150)]
    return {"test_providers": len(res), "base_fraud_rate": round(float(base), 4),
            "flagged": int(flag.sum()), "precision": round(prec, 4),
            "recall": round(rec, 4), "lift_over_base_rate": round(prec / base, 2),
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "triage": triage}


def make_plot(res_full, path):
    X, cols = build_feature_matrix(res_full)
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    coords = umap.UMAP(n_components=2, random_state=42).fit_transform(Xs)
    res_full = res_full.copy()
    res_full["ux"], res_full["uy"] = coords[:, 0], coords[:, 1]
    fig, ax = plt.subplots(figsize=(8, 6))
    nf = res_full[~res_full["flagged"]]; fl = res_full[res_full["flagged"]]
    ax.scatter(nf["ux"], nf["uy"], s=12, c="#c7d0d9", alpha=0.7,
               edgecolors="none", label="Not flagged")
    ax.scatter(fl["ux"], fl["uy"], s=28, c="#d1495b", alpha=0.9,
               edgecolors="white", linewidths=0.4, label="Flagged anomaly")
    ax.set_title("Provider behavioral landscape (scale-invariant features, 2D UMAP)",
                 fontsize=12, weight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(loc="best"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    print("[1/5] Loading data...")
    ip, op, bene, labels, source = load_data()
    print(f"      source = {source}, {len(labels)} providers")

    print("[2/5] Building scale-invariant + time-series features...")
    feat = build_provider_features(ip, op, bene, labels)

    print("[3/5] Unsupervised detection, OUT-OF-SAMPLE holdout...")
    res_test, _ = detect_with_holdout(feat)
    metrics = evaluate(res_test)

    print("[4/5] Supervised baseline + stability...")
    sup = supervised_baseline(feat)
    stab = stability(feat)

    print("[5/5] Scoring all providers for dashboard + writing outputs...")
    res_full, _ = detect_full(feat)
    res_full.sort_values("anomaly_score", ascending=False).to_csv(
        os.path.join(OUT, "scored_providers.csv"))
    make_plot(res_full, os.path.join(OUT, "provider_map.png"))

    report = {"data_source": source,
              "unsupervised_out_of_sample": metrics,
              "supervised_baseline": sup,
              "stability_across_seeds": stab}
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    m = metrics
    print("\n" + "=" * 56)
    print(" UNSUPERVISED  (out-of-sample held-out test providers)")
    print("=" * 56)
    print(f" test providers..... {m['test_providers']}")
    print(f" base fraud rate.... {m['base_fraud_rate']:.1%}")
    print(f" flagged (top 10%).. {m['flagged']}")
    print(f" precision.......... {m['precision']:.1%}")
    print(f" recall............. {m['recall']:.1%}")
    print(f" LIFT............... {m['lift_over_base_rate']}x")
    print(f" stability (5 seeds) {stab['lift_mean']}x +/- {stab['lift_std']}")
    print("-" * 56)
    print(" SUPERVISED baseline (RF, same features, OOS):")
    print(f"   AUC {sup['auc']}   precision@10% {sup['precision_at_10pct']:.0%}"
          f"   lift {sup['lift_at_10pct']}x")
    print("=" * 56)
    print(f"\nOutputs in {OUT}/")


if __name__ == "__main__":
    main()
