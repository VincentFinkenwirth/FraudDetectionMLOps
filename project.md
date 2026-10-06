# Task 3: Fraud detection in a government agency

(This file used to describe an older logistic-regression demo with generated data.
The project now uses the BAF dataset and CatBoost; the full description is in
[README.md](README.md).)

## Short version

```text
applications --> REST API (FastAPI) --> fraud probability + review flag
                     |      ^
                     v      | hot reload of models/current.json
               PostgreSQL   |
      (predictions, labels, |
       pipeline events)     |
                     |      |
                     v      |
     Jenkins: fraud-monitor (weekly) --> fraud-retrain (monthly / drift / performance)
                     |                        |
                     v                        v
                   MLflow (metrics, runs, model registry)
```

- Data: BAF `Base.csv` (NeurIPS 2022), split in time order. Real benefit data would be private.
- Model: CatBoost tuned with Optuna for maximum recall at FPR < 5 %. The scores are
  corrected for the class weights so the API returns real probabilities.
- The score only prioritises manual reviews; it does not decide eligibility.
- One simulated year (12 months x 4 weeks) with covariate shift, prior shift, an upstream
  scale change, concept drift and a data-quality incident. Labels arrive 4 weeks late.
- Jenkins retrains every 28 days, or earlier after data drift or a performance drop,
  and only deploys a candidate that passes the promotion gate. The API stays reachable
  during every deployment and is smoke-tested afterwards.

## What I would improve next

A real agency would need a privacy review, encrypted storage, a proper audit trail,
fairness work (the model flags applicants aged 50+ four times as often), a way to get
labels for applications that were not reviewed, and decisions made by trained staff.
