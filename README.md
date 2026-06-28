# Medicare Claims Anomaly Detection

An unsupervised screen that surfaces healthcare providers whose Medicare billing
patterns deviate from their peers, for payment-integrity review. Built as a proof
of concept for Cotiviti's intern screening (Topic 2: clustering and anomaly
detection for Treatment, Payment, and Operations).

The method extends an approach I built earlier for clinical-text analysis
(ClinsightAI): density-based outlier detection. There it flagged unusual patient
reviews as low-density noise points; here it flags providers whose billing
behavior does not look like their peers.

## What it does

1. Aggregates raw Medicare inpatient/outpatient claims to one row per provider.
2. Engineers scale-invariant behavioral features (per-claim, per-patient,
   per-physician ratios) plus monthly time-series features, so the model detects
   abnormal *behavior* rather than just large providers.
3. Scores each provider with HDBSCAN (cohort/outlier discovery) and Isolation
   Forest (continuous ranking), combined by percentile-rank averaging.
4. Surfaces a ranked review queue and explains each flagged provider in plain
   language via an LLM (LLaMA 3.1 on Groq), with a no-key template fallback.

## Results (out-of-sample, held-out test providers)

| Metric | Value |
|---|---|
| Lift over base rate | 2.1x (2.14x ± 0.30 across seeds) |
| Recall at top 10% | 43% |
| Base fraud rate | 9.3% |
| Supervised RF baseline (for comparison) | AUC 0.94, 6.3x lift |

The unsupervised number is intentionally modest and honest: removing a provider-size
confound and evaluating out of sample dropped lift from an inflated 4.3x to 2.1x.
See LIMITATIONS.md for the full account, including why an unsupervised screen is the
right tool despite the supervised baseline scoring higher on this labeled data.

## Run it

```bash
pip install -r requirements.txt

# place the 4 Kaggle "Healthcare Provider Fraud Detection" Train CSVs in data/raw/
#   Train-*.csv, Train_Inpatientdata-*.csv, Train_Outpatientdata-*.csv, Train_Beneficiarydata-*.csv

python run_pipeline.py        # builds features, scores providers, writes outputs/

# optional: plain-language explanations
export GROQ_API_KEY=your_key  # falls back to a template if unset
python explain.py

# dashboard
python app.py                 # then open http://localhost:5000
```

Without the Kaggle files, the loader generates a synthetic stand-in so the
pipeline still runs end to end.

## Layout

```
pipeline/
  data_loader.py   load real Kaggle CSVs (or synthetic fallback)
  features.py      provider-level scale-invariant + time-series features
  anomaly.py       HDBSCAN + Isolation Forest, train/test holdout, ranking
  supervised.py    Random Forest baseline (the "why not supervised" answer)
run_pipeline.py    orchestrates the pipeline, writes outputs/
explain.py         LLM (Groq) / template explanation for a flagged provider
app.py             Flask dashboard (review queue + case file)
LIMITATIONS.md     honest account of what this does and does not establish
```

## Data

Kaggle "Healthcare Provider Fraud Detection Analysis" dataset. The raw CSVs are
not committed (see .gitignore); download them and place the Train files in data/raw/.

## Note on the label

The dataset's `PotentialFraud` flag is a provided label of unknown adjudication,
not confirmed fraud. Reported metrics measure agreement with that label. This and
other limitations (specialty peer-grouping, claim-line granularity, scale,
governance) are documented in LIMITATIONS.md.

## Video Walkthrough
[Watch the demo](https://drive.google.com/file/d/1mfEfjDcNY89nCplHud8n7dJ-Rptkp3eK/view?usp=sharing)
