# Results of the simulated year

Final runs on 29 September 2026 with the second version of the pipeline (section 1):

- **All labels:** through the Docker Compose stack with Jenkins as the MLOps platform
  (`python -m simulation.replay --pipeline jenkins`: 52 `fraud-monitor` builds, 12 `fraud-retrain`
  builds with training).
- **Realistic labels:** three runs with different random audit samples
  (`bash work/runs/run_local.sh realistic_a 8101 realistic`, seeds 1 and 2 for b and c), each with its
  own API, database and models folder.

52 simulated weeks: 48 weeks of applications (**229,000** in total, 1.28 % fraud; 16,000 per month
before the downturn, 20,000 during it) plus 4 weeks in which only the last labels arrive. Labels
always arrive 4 weeks late. Tables: [docs/results/](docs/results/), charts:
[docs/results/charts/](docs/results/charts/) (one chart per month in `charts/months/`) and
[docs/guide/](docs/guide/). The numbers of the **first version** (the ones in the current slides)
are in section 8.

## 1. What changed after the first runs

| Change | Why (what the first runs showed) |
| --- | --- |
| **Fast loop:** weekly label-free workload threshold (5.5 % of last week's applications) | After the credit-bureau change the old threshold sent 12–19 % to review for 8 weeks; the fix mainly came from a new threshold, which needs no labels |
| **Evidence rule:** recall only judged with ≥ 20 labelled missed frauds (monitor) / ≥ 10 (gate) | In one realistic week the monitor reported 100 % recall (true: 44 %) because no audited fraud was in the window |
| **Gate at equal workload**, FPR check only with the old threshold policy, sanity check (AP ≥ 2× fraud rate) | Thresholds chosen on ~570 weighted rows made candidates look bad; at equal workload the FPR says nothing new |
| **Known-fraud check** in the gate | A candidate that caught 24 % instead of 75 % of the reviewed frauds was promoted (see section 7) |
| **Fixed tree count (377)** instead of early stopping | With realistic labels early stopping kept 1–2 trees in a third of the retrainings |
| **API reload + confirmation** after a threshold change | In fast local runs only 2 of ~45 threshold changes reached the API in time |
| **Rising volume** (16,000 → 20,000 per month) | In the first version the downturn made the months *smaller* (30,000 → 17,900) |
| **Logistic-regression baseline**, **three audit seeds** | To justify CatBoost with a number and to see whether one realistic run is luck |

## 2. Model before deployment

| Problem | Fix | Effect (validation) |
| --- | --- | --- |
| Scores were not probabilities (class weights inflate odds ~10x) | Prior correction in `scoring.py` | Mean score 8.5 % → 1.4 % (fraud rate 0.9 %); Brier 0.0273 → 0.0084; same applications flagged |
| Training only worked on my GPU | `configs/training_cpu.json`, `train.py --cpu` | Jenkins retrains on the CPU |
| No fairness check | FPR by age group in every evaluation | Applicants 50+: FPR 13.1 %, younger: 3.1 % (ratio 0.23) |
| No drift reference | `reference_profile.json` in every bundle | Monitoring compares with the data each model was accepted on |

Baseline (`baseline.py`, same split, same threshold rule, test set):

| | Logistic regression | CatBoost v1 |
| --- | ---: | ---: |
| Recall at FPR < 5 % | 53.1 % | **57.6 %** |
| Average precision | 0.144 | **0.177** |
| ROC AUC | 0.902 | **0.914** |

## 3. What the pipeline did (all labels)

| Decision (weekly monitoring) | Weeks |
| --- | ---: |
| no action | 26 |
| retrain | 13 |
| deferred — waiting for (drifted) labels | 9 |
| blocked / quality incident | 4 |

13 retraining decisions: **12 trainings, 10 promoted (v2–v11), 2 rejected**, 1 skipped (week 45:
not enough labelled data). The threshold was reset every week (except during the incident), the
API confirmed each change, and it was smoke-tested after every retraining.

| Week | Trigger | Result |
| ---: | --- | --- |
| 7 | schedule | **rejected**: candidate recall 3.2 pp lower |
| 11, 15, 19 | schedule | promoted (v2, v3, v4) |
| 22 | data drift + performance | promoted (v5) |
| 25 | data drift | promoted (v6) |
| 29, 32 | schedule / performance | promoted (v7, v8) |
| 35 | data drift | **rejected**: candidate recall 2.0 pp lower |
| 37–40 | — | **blocked** during the mobile-form incident |
| 41, 46, 50 | schedule (+ performance) | promoted (v9, v10, v11) |

## 4. How each injected change was detected

- **Downturn (months 3–12, rejection sampling + more applications):** drift share ≤ 0.034, so no
  data-drift alarm — as intended. The alert-rate signal fell (z = −5.8 in week 16).
- **New credit-bureau scale (from week 17):** found in the first week — PSI 8.25 on
  `credit_risk_score`, alert-rate z = +15.2. **Fast loop:** 16.9 % of the applications were sent to
  review in week 17 and 5.8 % in week 18 (first version: 12–16 % for 8 weeks). Retraining waited
  for labels of the new scale (deferred in weeks 17–18) and then updated the model in weeks 19,
  22 and 25.
- **Fraudsters' camouflage (months 7–12):** invisible to data monitoring (drift share ≤ 0.035),
  caught by the delayed-label performance trigger (weeks 32 and 38–41).
- **Mobile-form incident (month 10):** detected in all four weeks (PSI 2.4–2.6 on `device_os`,
  30 % unknown device type) → retraining blocked, threshold not changed, weeks quarantined.

![Weekly signals](docs/results/charts/weekly_signals.png)

## 5. The two loops: what does each one add? (all labels)

Same 229,000 applications, true labels:

| | Recall | FPR | Sent to review | Recall in top 5 % | Avg. precision | FPR ratio age |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Both loops** (retrained + workload threshold) | 48.0 % | 5.2 % | 5.8 % | **44.9 %** | **0.149** | 0.29 |
| Fast loop only (frozen v1 + workload threshold) | 47.2 % | 5.1 % | 5.7 % | 43.9 % | 0.138 | 0.30 |
| No MLOps (frozen v1, fixed threshold) | 55.5 % | 7.9 % | **8.5 %** | 43.9 % | 0.138 | 0.36 |

| Month | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FPR both loops | 4.9 | 4.8 | 5.0 | 4.4 | 7.8 | 4.8 | 5.2 | 5.5 | 5.1 | 5.4 | 4.7 | 5.0 |
| FPR frozen, fixed | 4.8 | 5.0 | 4.6 | 4.1 | 8.4 | 8.6 | 8.4 | 10.0 | 10.5 | 11.6 | 9.1 | 8.4 |
| Top-5 % recall both loops | 56.1 | 59.2 | 58.2 | 50.0 | 54.6 | 51.0 | 42.0 | 41.8 | 33.3 | 36.8 | 37.7 | 43.9 |
| Top-5 % recall fast loop only | 56.1 | 59.2 | 60.2 | 50.0 | 53.8 | 50.2 | 41.6 | 39.2 | 33.7 | 35.1 | 36.5 | 38.6 |

![Two loops](docs/guide/loops_ablation.png)

- The **fast loop** does most of the work: it keeps the review workload at the plan all year
  (5.7 % instead of 8.5 % — a third fewer reviews) and the FPR at 5.1 % instead of 7.9 %.
- **Retraining** adds a better ranking on top: +1.0 pp of the fraud in the top 5 % and 8 % higher
  average precision, mostly in the camouflage months (month 12: 43.9 % vs 38.6 %).
- The frozen model with its fixed threshold catches more fraud (55.5 %) only because it reviews
  50 % more applications.

## 6. Realistic labels: only reviewed and audited cases

Only **flagged** applications are reviewed and labelled (weight 1); a random **2 % audit** of the
others is labelled with weight 50; everything else stays unknown. Three runs with different audit
samples (the applications are identical):

| | Audit sample A | B | C | All labels |
| --- | ---: | ---: | ---: | ---: |
| Applications with a label | 7.6 % | 7.8 % | 7.6 % | 100 % |
| Extra reviews for the audit | 4,275 | 4,304 | 4,262 | – |
| Trainings / promoted / skipped | 11 / 11 / 3 | 12 / 12 / 0 | 11 / 10 / 3 | 12 / 10 / 1 |
| Weeks with measurable recall | 0 of 48 | 0 | 0 | 47 of 48 |
| Error of the recall estimate (mean / max, only shown) | 15 / 51 pp | 19 / 58 pp | 12 / 42 pp | 0 |
| Error of the FPR estimate (mean / max) | 0.25 / 1.2 pp | 0.23 / 1.1 pp | 0.22 / 0.6 pp | 0 |
| **True recall / FPR (year)** | 46.9 % / 5.2 % | 47.6 % / 5.3 % | 47.2 % / 5.2 % | 48.0 % / 5.2 % |
| Fraud in the top 5 % | 42.9 % | 43.7 % | 42.8 % | 44.9 % |
| Average precision | 0.130 | 0.132 | 0.132 | 0.149 |

(Fast loop only, for comparison: 47.2 % / 5.1 %, top-5 % 43.9 %, AP 0.138.)

![Estimate vs truth](docs/results/charts/realistic_estimate_vs_truth.png)

What this shows:

- **The results do not depend on the audit sample** (recall 46.9–47.6 %).
- **Label-free monitoring and the fast loop work the same** as with all labels.
- **The FPR is measurable, recall is not:** the monitor never had 20 labelled missed frauds in
  28 days, so it (correctly) never judged recall; the weighted estimate would have been off by
  12–19 pp on average.
- **The gate cannot see gains:** in almost every retraining recall was not comparable and the newer
  model was promoted; the known-fraud check rejected one candidate (82 % vs 98 % of the reviewed
  frauds).
- **Retraining with realistic labels adds nothing** over the fast loop — the ranking is even slightly
  worse (top-5 % 42.8–43.7 % vs 43.9 %). The models learn mainly from cases the old model already
  selected; the fraud it misses is only seen through ~2 audited cases per month.
- **Lesson:** with realistic labels the value of MLOps comes from label-free control and honest
  measurement; better models would need more label information (larger or targeted audit, other
  label sources, champion/challenger reviews).

### Open issue: recall with realistic labels (not fixed in this project)

For the fourth version I raised the audit from 2 % to 5 % (weight 20). It is still not enough:
the rebuilt months contain 165 audited fraud cases in the whole year, about 14 per 4 weeks, and
only ~8 of them are frauds the model missed. Recall needs exactly these cases. The monitor wants
≥ 20 in 28 days, the gate ≥ 10 in its 14-day holdout, so with realistic labels:

- recall is never measured and the performance trigger never fires (schedule and drift still do);
- the promotion gate can only reject clearly broken models (sanity check and the known-fraud
  check, see below), it cannot show that a new model is better on the drifted data.

I decided to keep this logic and state the limit instead of patching it, because every fix I
checked either needs more labels or brings in a new bias:

| Idea | What I measured / found | Why not now |
| --- | --- | --- |
| Lower the minimum of 10 / 20 missed frauds | – | the decision would be pure noise (my first realistic run rejected a candidate on 0 vs 1 audited case) |
| Audit that prefers higher scores (same 5 % budget, inverse-probability weights) | recall error per 4 weeks only ±10 → ±8 pp | still not usable |
| Fixed, fully labelled reference set (`reserve.csv`, 588 frauds) as a regression test | catches broken models with ±1–1.5 pp, but has none of the drift (true recall 37–62 %, reserve recall 58 → 51 %) | favours models that keep the old patterns; would not change any decision of the last run |
| Rolling evaluation set (reserve + held-back audit rows) | ~14 audited frauds a month → still ~85 % old data after a year | costs the only unbiased training labels for missed fraud |
| **Review hit rate** (share of reviewed cases that are fraud) as a performance signal | measured exactly every week, ±1 pp per 4 weeks; falls from ~12 % to 8–10 % in the camouflage months although the fraud rate rises | next step for monitoring |
| **Champion/challenger in shadow mode**, reviewing the cases only one model flags | ~1 % extra reviews give ~10 labelled frauds per 4 weeks that the champion missed → a fair paired comparison | next step for the gate |

With all labels none of this is needed: the holdout has ~65 labelled missed frauds and the
monitor ~130, so both comparisons work (only the noise of about ±3–4 pp on a 14-day holdout
remains).

## 7. Problems I found on the way (and fixed)

1. **A wrong recall of 100 %:** week 18 of the first realistic run — no audited fraud in the window,
   all 117 fraud labels had weight 1, so even the effective sample size was 117. True recall: 44 %.
   → evidence rule.
2. **Early stopping on selectively labelled data:** in the first attempt of the second version,
   about a third of the realistic retrainings stopped after 1–2 trees. One 2-tree model (v6) was
   promoted because recall was "not comparable" and caught ~20 % of the fraud for six months (year:
   36.6 %, see `docs/results/failed_attempt_early_stopping/`). The same snapshot retrained with 377
   trees: holdout AP 0.212 instead of 0.043. → fixed tree count + known-fraud check.
3. **Threshold changes that never arrived:** the API reads `current.json` every 5 s; in the fast local
   runs a simulated week took ~8 s and only 2 of ~45 changes were applied in time. → the weekly job
   reloads the API and checks the served threshold.
4. **A useless FPR check:** at equal workload every model has almost the same FPR; the noisy weighted
   estimate rejected a good candidate (7.09 %). → FPR check only with the old threshold policy.

## 8. First version (numbers in the current slides)

Before the changes of section 1, with 30,000 applications per month shrinking to ~17,900 during the
downturn (247,435 in total) — not directly comparable. Tables in `docs/results/first_version/`.

| | All labels | Realistic labels (1 run) | Frozen v1 |
| --- | ---: | ---: | ---: |
| Trainings / deployed | 14 / 12 | 12 / 4 (6 skipped) | – |
| Recall / FPR (year) | 51.0 % / 6.3 % | 50.2 % / 6.8 % | 55.0 % / 7.2 % |
| Sent to review | 6.8 % | 7.3 % | 7.8 % |
| Fraud in the top 5 % | 45.3 % | 43.8 % | 45.2 % |
| After the scale change | 8 weeks until the FPR was back | FPR 18.5 % in month 6 | – |

## 9. Critical reflection

- **The biggest lever was not retraining but the threshold.** A label-free rule kept the workload
  and the FPR at the plan all year; retraining added a modest ranking gain with all labels and none
  with realistic labels.
- **Monitoring the right thing matters more than more rules:** three of the four problems in
  section 7 were measurement problems (a wrong estimate, a wrong early-stopping signal, an
  unconfirmed change), found only because the simulation knows the true labels.
- **Camouflage still hurts every variant:** recall falls from ~60 % to 35–45 %. The performance
  trigger reacts, but the retrained models can only partly relearn the copied signals.
- **Fairness does not fix itself:** the FPR ratio between age groups stayed at ~0.29 (0.36 for the
  frozen model, which flags more people).
- **Limits of the simulation:** only 4 real BAF months stretched to 12; labels from a perfect review
  after exactly 4 weeks; the realistic runs ran locally (same code as Jenkins — the first 8 weeks'
  thresholds were identical to the Jenkins run).
