# Build one row per provider. Detect on ratios, not raw counts, so the model
# doesn't just flag big hospitals. Added monthly burstiness for the time-series part.

import numpy as np
import pandas as pd

DIAG_COLS = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
PROC_COLS = [f"ClmProcedureCode_{i}" for i in range(1, 7)]

# scale-invariant features only (ratios/intensities, no raw counts)
DETECTION_FEATURES = [
    "dollars_per_claim",
    "reimb_cv",
    "reimb_max_to_mean",
    "claims_per_bene",
    "diagnoses_per_claim",
    "procedures_per_claim",
    "mean_los",
    "pct_inpatient",
    "claims_per_physician",
    "bene_per_physician",
    "deductible_ratio",
    "chronic_per_bene",
    "ts_burstiness",
    "ts_monthly_cv",
]


def _per_claim_counts(df):
    df = df.copy()
    if "n_diagnoses" not in df.columns:
        present = [c for c in DIAG_COLS if c in df.columns]
        df["n_diagnoses"] = df[present].notna().sum(axis=1) if present else 0
    if "n_procedures" not in df.columns:
        present = [c for c in PROC_COLS if c in df.columns]
        df["n_procedures"] = df[present].notna().sum(axis=1) if present else 0
    if "los_days" not in df.columns:
        if "AdmissionDt" in df.columns and "DischargeDt" in df.columns:
            adm = pd.to_datetime(df["AdmissionDt"], errors="coerce")
            dis = pd.to_datetime(df["DischargeDt"], errors="coerce")
            df["los_days"] = (dis - adm).dt.days.clip(lower=0)
        else:
            df["los_days"] = np.nan  # genuinely unknown for outpatient; handled later
    return df


def _timeseries_features(claims):
    """Per-provider monthly claim-volume series -> burstiness + variability."""
    if "ClaimStartDt" not in claims.columns:
        return pd.DataFrame(index=claims["Provider"].unique())

    c = claims.copy()
    c["month"] = pd.to_datetime(c["ClaimStartDt"], errors="coerce").dt.to_period("M")
    monthly = c.groupby(["Provider", "month"]).size().reset_index(name="n")

    def agg(grp):
        vals = grp["n"].values
        mean = vals.mean()
        return pd.Series({
            "ts_burstiness": vals.max() / mean if mean > 0 else 1.0,
            "ts_monthly_cv": vals.std() / mean if mean > 0 else 0.0,
        })

    return monthly.groupby("Provider").apply(agg)


def build_provider_features(inpatient, outpatient, beneficiary, labels):
    inpatient = _per_claim_counts(inpatient)
    outpatient = _per_claim_counts(outpatient)
    inpatient["is_inpatient"] = 1
    outpatient["is_inpatient"] = 0
    claims = pd.concat([inpatient, outpatient], ignore_index=True)

    if "DeductibleAmtPaid" not in claims.columns:
        claims["DeductibleAmtPaid"] = 0.0
    if "AttendingPhysician" not in claims.columns:
        claims["AttendingPhysician"] = "UNK"

    g = claims.groupby("Provider")

    raw = pd.DataFrame({
        "n_claims":            g.size(),
        "n_inpatient":         g["is_inpatient"].sum(),
        "mean_reimbursed":     g["InscClaimAmtReimbursed"].mean(),
        "std_reimbursed":      g["InscClaimAmtReimbursed"].std(),
        "max_reimbursed":      g["InscClaimAmtReimbursed"].max(),
        "mean_deductible":     g["DeductibleAmtPaid"].mean(),
        "mean_n_diagnoses":    g["n_diagnoses"].mean(),
        "mean_n_procedures":   g["n_procedures"].mean(),
        "mean_los":            g["los_days"].mean(),
        "n_unique_bene":       g["BeneID"].nunique(),
        "n_unique_attending":  g["AttendingPhysician"].nunique(),
    })

    feat = pd.DataFrame(index=raw.index)
    feat["dollars_per_claim"]    = raw["mean_reimbursed"]
    feat["reimb_cv"]             = raw["std_reimbursed"] / raw["mean_reimbursed"]
    feat["reimb_max_to_mean"]    = raw["max_reimbursed"] / raw["mean_reimbursed"]
    feat["claims_per_bene"]      = raw["n_claims"] / raw["n_unique_bene"].clip(lower=1)
    feat["diagnoses_per_claim"]  = raw["mean_n_diagnoses"]
    feat["procedures_per_claim"] = raw["mean_n_procedures"]
    feat["mean_los"]             = raw["mean_los"]
    feat["pct_inpatient"]        = raw["n_inpatient"] / raw["n_claims"].clip(lower=1)
    feat["claims_per_physician"] = raw["n_claims"] / raw["n_unique_attending"].clip(lower=1)
    feat["bene_per_physician"]   = raw["n_unique_bene"] / raw["n_unique_attending"].clip(lower=1)
    feat["deductible_ratio"]     = raw["mean_deductible"] / raw["mean_reimbursed"].clip(lower=1)

    bene = beneficiary.copy()
    if "chronic_count" not in bene.columns:
        chronic_cols = [c for c in bene.columns if c.lower().startswith("chroniccond")]
        if chronic_cols:
            bene["chronic_count"] = (bene[chronic_cols] == 1).sum(axis=1)
    if "chronic_count" in bene.columns:
        bene_chronic = bene.set_index("BeneID")["chronic_count"]
        claims["bene_chronic"] = claims["BeneID"].map(bene_chronic)
        feat["chronic_per_bene"] = claims.groupby("Provider")["bene_chronic"].mean()
    else:
        feat["chronic_per_bene"] = np.nan

    ts = _timeseries_features(claims)
    for col in ["ts_burstiness", "ts_monthly_cv"]:
        feat[col] = ts[col] if col in ts.columns else np.nan

    for col in ["n_claims", "n_unique_bene", "n_unique_attending",
                "max_reimbursed", "mean_reimbursed"]:
        feat[col + "_ctx"] = raw[col]

    label_map = labels.set_index("Provider")["PotentialFraud"]
    feat["PotentialFraud"] = feat.index.map(label_map).fillna("No")
    return feat


def build_feature_matrix(feat):
    # median-impute missing (don't conflate missing with 0)
    cols = [c for c in DETECTION_FEATURES if c in feat.columns]
    X = feat[cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    return X, cols


if __name__ == "__main__":
    from data_loader import load_data
    ip, op, bene, labels, source = load_data()
    feat = build_provider_features(ip, op, bene, labels)
    X, cols = build_feature_matrix(feat)
    print(f"Source: {source}")
    print(f"{feat.shape[0]} providers, {len(cols)} detection features")
    print("Detection features:", cols)
    print("\nSummary (detection features):")
    print(X.describe().T[["mean", "std", "min", "max"]].round(2).to_string())
