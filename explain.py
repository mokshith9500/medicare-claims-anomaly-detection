# Plain-English explanation for a flagged provider.
# The LLM only rewords the z-scores the pipeline already computed (it doesn't
# decide anything). Falls back to a template if there's no GROQ_API_KEY.

import os
import numpy as np
import pandas as pd

from pipeline.features import DETECTION_FEATURES as FEATURE_COLS

# human-readable labels + whether a HIGH value is the suspicious direction
FEATURE_INFO = {
    "dollars_per_claim":    ("average reimbursement per claim", "high"),
    "reimb_cv":             ("variability of reimbursement amounts", "high"),
    "reimb_max_to_mean":    ("largest claim relative to its own average", "high"),
    "claims_per_bene":      ("claims billed per unique patient", "high"),
    "diagnoses_per_claim":  ("diagnosis codes attached per claim", "high"),
    "procedures_per_claim": ("procedure codes attached per claim", "high"),
    "mean_los":             ("average inpatient length of stay", "high"),
    "pct_inpatient":        ("share of inpatient claims", "high"),
    "claims_per_physician": ("claims billed per physician", "high"),
    "bene_per_physician":   ("patients per physician", "high"),
    "deductible_ratio":     ("deductible as a share of reimbursement", "low"),
    "chronic_per_bene":     ("chronic conditions per patient (case mix)", "high"),
    "ts_burstiness":        ("billing concentrated in a peak month", "high"),
    "ts_monthly_cv":        ("month-to-month billing volatility", "high"),
}

GROQ_MODEL = "llama-3.1-8b-instant"   # or llama-3.3-70b-versatile for nicer text


def build_evidence(res, provider_id, top_n=4):
    # z-score this provider vs everyone, keep the most abnormal features
    cols = [c for c in FEATURE_COLS if c in res.columns]
    means = res[cols].mean()
    stds = res[cols].std().replace(0, 1e-9)

    row = res.loc[provider_id]
    z = (row[cols] - means) / stds

    # rank by how abnormal, keep the strongest deviations
    ranked = z.abs().sort_values(ascending=False).head(top_n)

    evidence = []
    for feat in ranked.index:
        label, suspicious_dir = FEATURE_INFO.get(feat, (feat, "high"))
        direction = "above" if z[feat] > 0 else "below"
        evidence.append({
            "feature": label,
            "provider_value": round(float(row[feat]), 2),
            "peer_average": round(float(means[feat]), 2),
            "std_devs": round(float(z[feat]), 1),
            "direction": direction,
            "is_suspicious_direction": (
                (z[feat] > 0 and suspicious_dir == "high") or
                (z[feat] < 0 and suspicious_dir == "low")
            ),
        })

    return {
        "provider_id": provider_id,
        "anomaly_score": round(float(row["anomaly_score"]), 3),
        "flagged_by_hdbscan": bool(row.get("hdbscan_noise", False)
                                   or row.get("small_cluster", False)),
        "evidence": evidence,
    }


def _template_explanation(ev):
    # no-LLM fallback: build the text straight from the numbers
    lines = [
        f"Provider {ev['provider_id']} was flagged with an anomaly score of "
        f"{ev['anomaly_score']} (top decile of risk). The billing profile "
        f"departs from peer norms on several measures:"
    ]
    for e in ev["evidence"]:
        flag = " (elevated vs peers)" if e["is_suspicious_direction"] else ""
        lines.append(
            f"  - {e['feature']} is {abs(e['std_devs'])} standard deviations "
            f"{e['direction']} the peer average "
            f"({e['provider_value']} vs {e['peer_average']}){flag}."
        )
    lines.append(
        "Recommended next step: pull a sample of this provider's claims and "
        "verify documentation supports the billed amounts, code counts, and "
        "patient volume. Note that high values can reflect a legitimate "
        "specialty or case mix, so this is a screening signal for review, not "
        "a determination of fraud."
    )
    return "\n".join(lines)


def _llm_explanation(ev):
    # narrate the evidence with Groq; raises on failure so caller can fall back
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    facts = "\n".join(
        f"- {e['feature']}: {e['provider_value']} vs peer average "
        f"{e['peer_average']} ({abs(e['std_devs'])} SD {e['direction']})"
        for e in ev["evidence"]
    )
    system = (
        "You are a healthcare payment-integrity analyst assistant. You explain "
        "why a provider was flagged for review using ONLY the evidence given. "
        "Never invent numbers. Be concise and factual. Never state the provider "
        "is guilty; say the pattern 'warrants review'. End with a short, "
        "concrete checklist of what a human reviewer should verify."
    )
    user = (
        f"Provider {ev['provider_id']} was flagged by an unsupervised anomaly "
        f"detector (score {ev['anomaly_score']}, top decile of risk).\n\n"
        f"Evidence (how this provider deviates from peers):\n{facts}\n\n"
        f"Write a 3-4 sentence explanation followed by a 3-item checklist."
    )

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


def explain_provider(res, provider_id, top_n=4):
    # returns (text, "llm"|"template")
    ev = build_evidence(res, provider_id, top_n=top_n)
    if os.environ.get("GROQ_API_KEY"):
        try:
            return _llm_explanation(ev), "llm"
        except Exception as e:
            print(f"[explain] LLM call failed ({e}); using template fallback.")
    return _template_explanation(ev), "template"


if __name__ == "__main__":
    from pipeline.data_loader import load_data
    from pipeline.features import build_provider_features
    from pipeline.anomaly import detect_full

    ip, op, bene, labels, source = load_data()
    feat = build_provider_features(ip, op, bene, labels)
    res, _ = detect_full(feat)

    # explain the single highest-risk provider
    top_id = res.sort_values("anomaly_score", ascending=False).index[0]
    text, used = explain_provider(res, top_id)

    print(f"[data source: {source}]  [explanation source: {used}]")
    print(f"[true label for this provider: {res.loc[top_id, 'PotentialFraud']}]\n")
    print(text)
