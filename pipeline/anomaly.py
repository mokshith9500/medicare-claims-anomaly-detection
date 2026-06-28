# Unsupervised detection. Fit on train, score held-out test so the numbers are
# honest. HDBSCAN + Isolation Forest, combined by rank averaging.

import numpy as np
import pandas as pd
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from .features import build_feature_matrix


def _rank01(x):
    # values -> percentile ranks in [0,1]
    return pd.Series(x).rank(pct=True).values


def fit_detectors(X_train, contamination=0.10, min_cluster_size=15, seed=42):
    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)

    iso = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=seed).fit(Xs)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5,
                                metric="euclidean", prediction_data=True).fit(Xs)
    # small clusters = possible billing rings
    sizes = pd.Series(clusterer.labels_).value_counts()
    small = {c for c, n in sizes.items()
             if c != -1 and n < 0.08 * len(X_train)}
    return {"scaler": scaler, "iso": iso, "clusterer": clusterer,
            "small_clusters": small}


def score(models, X, flag_top_pct=0.10):
    # score new providers with fitted models
    Xs = models["scaler"].transform(X)

    # isolation forest score (higher = weirder)
    if_raw = -models["iso"].decision_function(Xs)

    # hdbscan: low membership strength = outlier
    labels, strengths = hdbscan.approximate_predict(models["clusterer"], Xs)
    hdb_outlier = 1.0 - strengths
    is_noise = (labels == -1)
    is_small = np.isin(labels, list(models["small_clusters"]))

    # combine the two by averaging their percentile ranks
    score_combined = (_rank01(if_raw) + _rank01(hdb_outlier)) / 2.0

    out = pd.DataFrame(index=X.index)
    out["iforest_score"] = _rank01(if_raw)
    out["hdbscan_outlier"] = _rank01(hdb_outlier)
    out["hdbscan_noise"] = is_noise
    out["small_cluster"] = is_small
    out["anomaly_score"] = score_combined
    threshold = pd.Series(score_combined).quantile(1 - flag_top_pct)
    out["flagged"] = out["anomaly_score"] >= threshold
    return out


def detect_with_holdout(feat, test_size=0.30, contamination=0.10,
                        flag_top_pct=0.10, seed=42):
    # fit on train, score held-out test (this is the honest evaluation)
    X, cols = build_feature_matrix(feat)
    y = (feat["PotentialFraud"] == "Yes")

    X_tr, X_te = train_test_split(X, test_size=test_size,
                                  random_state=seed, stratify=y)
    models = fit_detectors(X_tr, contamination=contamination, seed=seed)
    scored = score(models, X_te, flag_top_pct=flag_top_pct)

    res = feat.loc[X_te.index].join(scored)
    return res, models


def detect_full(feat, contamination=0.10, flag_top_pct=0.10, seed=42):
    # fit+score on everyone, just so the dashboard can show all providers
    X, cols = build_feature_matrix(feat)
    models = fit_detectors(X, contamination=contamination, seed=seed)
    scored = score(models, X, flag_top_pct=flag_top_pct)
    return feat.join(scored), models


def stability(feat, seeds=(0, 1, 2, 3, 4), flag_top_pct=0.10):
    # re-run across seeds to check the lift isn't a fluke
    lifts = []
    for s in seeds:
        res, _ = detect_with_holdout(feat, seed=s, flag_top_pct=flag_top_pct)
        y = (res["PotentialFraud"] == "Yes")
        base = y.mean()
        prec = (res.loc[res["flagged"], "PotentialFraud"] == "Yes").mean()
        lifts.append(prec / base if base else 0)
    return {"lift_mean": round(float(np.mean(lifts)), 2),
            "lift_std": round(float(np.std(lifts)), 2),
            "lifts": [round(x, 2) for x in lifts]}


if __name__ == "__main__":
    from data_loader import load_data
    from features import build_provider_features

    ip, op, bene, labels, source = load_data()
    feat = build_provider_features(ip, op, bene, labels)

    res, _ = detect_with_holdout(feat)
    y = (res["PotentialFraud"] == "Yes")
    base = y.mean()
    flagged = res["flagged"]
    tp = int((flagged & y).sum()); fp = int((flagged & ~y).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / y.sum() if y.sum() else 0
    print(f"[source: {source}]  OUT-OF-SAMPLE (held-out test providers)")
    print(f"test providers: {len(res)}   base fraud rate: {base:.1%}")
    print(f"precision: {prec:.1%}   recall: {rec:.1%}   lift: {prec/base:.2f}x")
