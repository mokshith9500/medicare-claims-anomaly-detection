# Limitations and Path to Production

This is a two-day proof of concept, not a production fraud system. Below is an
honest account of what it does and does not establish, written so a reviewer can
see the engineering judgment behind it. Some critiques were fixed in code (v2);
others are inherent and are named here rather than hidden.

## Fixed in v2 (after self-review)

- **Size confound.** v1 detected partly on raw counts and largely flagged *large*
  providers. v2 detects only on scale-invariant ratios (per-claim, per-patient,
  per-physician). Honest lift dropped from 4.3x (in-sample, size-confounded) to
  ~2.1x (out-of-sample, behavioral). The drop is the point.
- **In-sample evaluation.** v2 uses a train/test split: detectors fit on train,
  metrics reported on held-out providers.
- **Missing time-series.** v2 adds monthly burstiness and volatility features.
- **No supervised comparison.** v2 adds a Random Forest baseline (AUC ~0.94,
  ~6.3x lift) to quantify the gap and justify the unsupervised choice.
- **Hand-tuned score blend.** Replaced with percentile-rank averaging.
- **fillna(0).** Replaced with median imputation.
- **Stability unknown.** v2 reports lift across 5 seeds (~2.1x +/- 0.3).

## Inherent limitations (not fixed; would need real production work)

1. **Label provenance.** Kaggle "PotentialFraud" is a provided label of unknown
   adjudication, not confirmed fraud. Metrics measure agreement with that label,
   not ground-truth fraud.
2. **No specialty peer-grouping.** A cardiologist should be compared to
   cardiologists. The dataset lacks specialty, so high-cost specialists are a
   real false-positive source (see the demo: the top-flagged provider is a
   legitimate high-intensity provider). Production needs specialty + geography
   risk adjustment.
3. **Provider-level aggregation.** Averaging hides a fraudster who bills mostly
   clean claims with a few bad ones. Production operates at the claim-line level.
4. **Evadability.** Outlier detection catches the unsophisticated. A fraudster who
   bills near the mean evades it. This is inherent to anomaly methods.
5. **Scale.** This is in-memory pandas on ~0.5M claims. Real Medicare is billions
   of lines and needs distributed compute (Spark) and a feature store.
6. **Governance.** No access control, audit trail, HIPAA controls, or
   adverse-action explainability. The LLM layer narrates evidence only and must
   never drive an automated adverse decision in a regulated setting.
7. **Concept drift.** Fraud evolves; the model is static. Production needs
   monitoring and a retraining cadence.

## What this POC legitimately demonstrates

A working, validated, out-of-sample unsupervised screen that concentrates known
fraud ~2x over the base rate using only behavioral (non-size) signals, paired
with a transparent GenAI explanation layer and a supervised baseline that frames
the tradeoff. It proves the concept and shows a clear, honest path to production.
