# Flask dashboard for the claims anomaly screen.
# Loads the scored providers from the last run_pipeline.py run and serves a
# triage console: ranked worklist on the left, case file on the right.

import json
import os
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from pipeline.features import DETECTION_FEATURES
from explain import explain_provider, FEATURE_INFO

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "outputs")

app = Flask(__name__)

# load once at startup
DF = pd.read_csv(os.path.join(OUT, "scored_providers.csv"), index_col=0)
MEANS = DF[DETECTION_FEATURES].mean()
STDS = DF[DETECTION_FEATURES].std().replace(0, 1e-9)
try:
    REPORT = json.load(open(os.path.join(OUT, "report.json")))
except FileNotFoundError:
    REPORT = {}


def zscores(pid):
    row = DF.loc[pid]
    z = ((row[DETECTION_FEATURES] - MEANS) / STDS)
    out = []
    for f in DETECTION_FEATURES:
        label = FEATURE_INFO.get(f, (f, "high"))[0]
        out.append({"feature": label, "z": round(float(z[f]), 2),
                    "value": round(float(row[f]), 2),
                    "peer": round(float(MEANS[f]), 2)})
    out.sort(key=lambda d: abs(d["z"]), reverse=True)
    return out


@app.route("/")
def index():
    m = REPORT.get("unsupervised_out_of_sample", {})
    s = REPORT.get("supervised_baseline", {})
    return render_template("index.html",
                           lift=m.get("lift_over_base_rate", "-"),
                           base=round(m.get("base_fraud_rate", 0) * 100, 1),
                           sup_lift=s.get("lift_at_10pct", "-"),
                           n=len(DF))


@app.route("/api/providers")
def providers():
    top = DF.sort_values("anomaly_score", ascending=False).head(150)
    rows = [{
        "id": pid,
        "score": round(float(r["anomaly_score"]), 3),
        "flagged": bool(r["flagged"]),
        "label": r.get("PotentialFraud", "No"),
        "dollars": int(r.get("dollars_per_claim", 0)),
        "claims": int(r.get("n_claims_ctx", 0)),
    } for pid, r in top.iterrows()]
    return jsonify(rows)


@app.route("/api/provider/<pid>")
def provider(pid):
    if pid not in DF.index:
        return jsonify({"error": "not found"}), 404
    r = DF.loc[pid]
    return jsonify({
        "id": pid,
        "score": round(float(r["anomaly_score"]), 3),
        "flagged": bool(r["flagged"]),
        "label": r.get("PotentialFraud", "No"),
        "deviations": zscores(pid),
    })


@app.route("/api/explain/<pid>", methods=["POST"])
def explain(pid):
    if pid not in DF.index:
        return jsonify({"error": "not found"}), 404
    text, source = explain_provider(DF, pid)
    return jsonify({"text": text, "source": source})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
