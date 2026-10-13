# Prepared data for reproduction

This folder contains the data needed to run the supplied fraud service, Jenkins training smoke test, and simulated year without downloading the original CSV separately.

## Source and attribution

The source is **Bank Account Fraud Dataset Suite (BAF), Base.csv**, by Sergio Jesus, Jose Pombal, Duarte Alves, Andre F. Cruz, Pedro Saleiro, Rita P. Ribeiro, Joao Gama, and Pedro Bizarro (Feedzai).

Paper: *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation*, NeurIPS 2022.

- [Source dataset on Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)
- [Authors' project and citation](https://github.com/feedzai/bank-account-fraud)
- [Authors' dataset datasheet](https://github.com/feedzai/bank-account-fraud/blob/main/documents/datasheet.pdf)

The SHA-256 of the original `Base.csv` is:

```text
7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809
```

The original 203.5 MiB CSV is excluded from Git. The prepared files below are derived from it and retain their upstream data terms.

## Upstream license notices

The Kaggle listing's public API reports **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**, checked on 13 October 2026. See the [license](https://creativecommons.org/licenses/by-nc-sa/4.0/) and [legal code](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en).

The authors' separate datasheet, question A44, states **CC BY-NC-ND 4.0**, which differs on derivative sharing. Both notices are recorded here rather than treating the repository's code license as a data license. Clarify the applicable notice with the dataset publisher before publishing the modified simulation files if the conflicting notices have not been resolved. This document does not grant additional rights.

## Included files and changes

| Path | Rows | Preparation |
| --- | ---: | --- |
| `splits/train.csv` | 400,000 | First consecutive source block |
| `splits/val.csv` | 100,000 | Next consecutive source block |
| `splits/test.csv` | 100,000 | Next consecutive source block |
| `splits/simulation_pool.csv` | 360,000 | Next consecutive source block |
| `splits/reserve.csv` | 40,000 | Final consecutive source block |
| `splits/split_metadata.json` | - | Source hash, split boundaries, counts, and provenance |
| `simulation/applications/month_01.csv` through `month_12.csv` | 286,300 total | Prepared application batches without fraud labels |
| `simulation/labels/month_01.csv` through `month_12.csv` | 286,300 total | Matching outcomes and simulated availability times |
| `simulation/summary.json` | - | Scenario, monthly counts, and injected changes |

These are prepared project files, not an unchanged redistribution of the complete source dataset:

- `training/prepare_data.py` preserves source order and adds `source_id` to identify the original CSV row.
- `simulation/build_months.py` filters applications to simulate population and fraud-prevalence shifts, offsets credit scores by 600, copies selected legitimate features into some fraud cases, and injects missing session lengths and an unseen device category during the quality incident.
- The generator adds application IDs, timestamps, audit selections, and delayed label availability. Model inputs and private simulation labels are stored separately.
- The scenario is defined in `configs/drift_scenario.json`, with seed 2026 and a four-week label delay. Every month uses previously unused source rows.

Only the listed prepared files are allowed by `.gitignore`. Raw downloads, local databases, runtime model versions, credentials, and unrelated data remain excluded.
