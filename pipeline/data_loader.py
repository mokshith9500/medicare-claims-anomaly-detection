# Loads the Kaggle claims CSVs from data/raw/. If they're not there, makes a
# synthetic stand-in so the pipeline still runs while developing.

import glob
import os
import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def _find(pattern, prefer="train"):
    """Return a file in data/raw matching a substring, preferring train over test."""
    matches = [f for f in glob.glob(os.path.join(RAW_DIR, "*.csv"))
               if pattern.lower() in os.path.basename(f).lower()]
    if not matches:
        return None
    preferred = [f for f in matches if prefer in os.path.basename(f).lower()]
    return preferred[0] if preferred else matches[0]


def load_real():
    """
    Load the real Kaggle CSVs if all required tables are present.
    Returns (inpatient, outpatient, beneficiary, labels) or None if not found.
    """
    ip = _find("Inpatient")
    op = _find("Outpatient")
    bene = _find("Beneficiary")
    # provider label file: has no 'data' tag, prefer the train labels
    label = None
    label_matches = []
    for f in glob.glob(os.path.join(RAW_DIR, "*.csv")):
        name = os.path.basename(f).lower()
        if "data" not in name:
            label_matches.append(f)
    train_labels = [f for f in label_matches
                    if "train" in os.path.basename(f).lower()]
    if train_labels:
        label = train_labels[0]
    elif label_matches:
        label = label_matches[0]

    if not all([ip, op, bene, label]):
        return None

    return (
        pd.read_csv(ip),
        pd.read_csv(op),
        pd.read_csv(bene),
        pd.read_csv(label),
    )


def generate_synthetic(n_providers=1000, seed=42):
    """
    Generate a synthetic dataset matching the columns our feature step uses.
    ~9% of providers are marked fraudulent and given an exaggerated billing
    pattern so the anomaly detector has a real signal to find.
    """
    rng = np.random.default_rng(seed)

    provider_ids = [f"PRV{100000 + i}" for i in range(n_providers)]
    is_fraud = rng.random(n_providers) < 0.09  # base fraud rate ~9%

    # real fraud is heterogeneous: different schemes leave different fingerprints.
    # assign each fraud provider one of three archetypes so anomalies scatter
    # instead of forming one tidy cluster.
    #   inflation - bills far higher dollar amounts
    #   phantom   - reuses a tiny pool of beneficiaries at high volume
    #   upcoding  - attaches abnormally many diagnosis/procedure codes
    schemes = np.where(
        is_fraud,
        rng.choice(["inflation", "phantom", "upcoding"], size=n_providers),
        "none",
    )

    ip_rows, op_rows = [], []
    bene_rows = {}
    claim_counter = 0

    for pid, fraud, scheme in zip(provider_ids, is_fraud, schemes):
        # claim volume: phantom schemes bill more; others modestly elevated
        vol_mult = 1.8 if scheme == "phantom" else (1.2 if fraud else 1.0)
        n_claims = int(rng.integers(15, 60) * vol_mult)

        # beneficiary reuse: phantom schemes lean on a smaller patient pool
        reuse = 0.25 if scheme == "phantom" else (0.65 if fraud else 0.8)
        n_bene_pool = max(3, int(n_claims * reuse))
        bene_pool = [f"BENE{rng.integers(1, 9_000_000)}" for _ in range(n_bene_pool)]

        # provider-level jitter, and a few honest providers naturally look odd
        # (this creates realistic overlap -> false positives, believable metrics)
        amt_jitter = rng.uniform(0.8, 1.3)
        if not fraud and rng.random() < 0.05:
            amt_jitter *= rng.uniform(1.3, 1.8)  # unusual but legitimate provider

        # temporal pattern: fraud schemes often bill in bursts (few active months);
        # honest providers spread claims across the year
        if fraud and rng.random() < 0.6:
            active_months = rng.choice(range(1, 13),
                                       size=rng.integers(2, 5), replace=False)
        else:
            active_months = np.arange(1, 13)

        for _ in range(n_claims):
            claim_counter += 1
            bid = bene_pool[rng.integers(0, n_bene_pool)]
            inpatient = rng.random() < (0.38 if fraud else 0.30)

            base = rng.lognormal(mean=8.2 if inpatient else 6.6, sigma=0.7)
            amt = base * (1.6 * amt_jitter if scheme == "inflation" else amt_jitter)

            n_diag = int(rng.integers(1, 6) + (3 if scheme == "upcoding" else 0))
            n_proc = int(rng.integers(0, 3) + (2 if scheme == "upcoding" else 0))

            los = int(rng.integers(1, 6) +
                      (3 if scheme == "upcoding" else 0)) if inpatient else 0

            attending = f"PHY{rng.integers(1, 80000)}"

            month = int(active_months[rng.integers(0, len(active_months))])
            day = int(rng.integers(1, 28))
            claim_date = f"2009-{month:02d}-{day:02d}"

            row = {
                "Provider": pid,
                "ClaimID": f"CLM{claim_counter}",
                "BeneID": bid,
                "ClaimStartDt": claim_date,
                "InscClaimAmtReimbursed": round(amt, 2),
                "DeductibleAmtPaid": round(amt * rng.uniform(0.0, 0.12), 2),
                "AttendingPhysician": attending,
                "n_diagnoses": n_diag,
                "n_procedures": n_proc,
                "los_days": los,
            }
            (ip_rows if inpatient else op_rows).append(row)

            if bid not in bene_rows:
                # rough age 35-95 via a birth year
                bene_rows[bid] = {
                    "BeneID": bid,
                    "birth_year": int(rng.integers(1930, 1990)),
                    "chronic_count": int(rng.integers(0, 11)),
                }

    inpatient = pd.DataFrame(ip_rows)
    outpatient = pd.DataFrame(op_rows)
    beneficiary = pd.DataFrame(list(bene_rows.values()))
    labels = pd.DataFrame(
        {"Provider": provider_ids,
         "PotentialFraud": np.where(is_fraud, "Yes", "No")}
    )
    return inpatient, outpatient, beneficiary, labels


def load_data():
    """
    Main entry point. Returns (inpatient, outpatient, beneficiary, labels, source)
    where source is 'real' or 'synthetic'.
    """
    real = load_real()
    if real is not None:
        ip, op, bene, labels = real
        return ip, op, bene, labels, "real"

    ip, op, bene, labels = generate_synthetic()
    return ip, op, bene, labels, "synthetic"


if __name__ == "__main__":
    ip, op, bene, labels, source = load_data()
    print(f"Source: {source}")
    print(f"Inpatient claims:  {len(ip):,}")
    print(f"Outpatient claims: {len(op):,}")
    print(f"Beneficiaries:     {len(bene):,}")
    print(f"Providers:         {len(labels):,}")
    print(f"Fraud rate:        {(labels.PotentialFraud == 'Yes').mean():.1%}")
