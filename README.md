# Amazon ML Challenge 2026: Business Entity Resolution

## Project Overview
This repository contains the baseline implementation and outputs for the Amazon Business Entity Resolution Challenge.

- **Current Leaderboard Score:** $F_{0.5} = 0.866$
- **Baseline Git Tag:** `baseline-0.866`

---

## Repository Structure

```
Amazon-ML-Final/
│
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # Source code (blocking, features, training, inference)
│       ├── README.md               # Pipeline documentation & instructions
│       └── requirements.txt        # Pinned python dependencies
│
├── output/
│   ├── matching_results.tsv        # Baseline submission file (F_0.5 = 0.866)
│   └── candidate_pairs.tsv         # Candidate pairs evaluated by final matching model
│
├── student_resource/
│   ├── utils/
│   │   └── validate_submission.py  # Official submission format validation script
│   └── README.md                   # Problem statement & challenge rules
│
├── Documentation_template.md       # Solution methodology document
├── .gitignore                      # Excludes raw multi-GB datasets & intermediate caches
└── README.md                       # Project baseline summary
```

---

## Operational Workflow

* **Baseline Preservation:** This repository reflects the exact baseline state that achieved the public leaderboard score of $F_{0.5} = 0.866$.
* **Execution Environment:**
  * Heavy training and large-scale dense retrieval inference are designed to run on high-compute / GPU environments (e.g., Kaggle / cloud compute nodes).
  * The local development environment is used for code development, feature engineering validation, and controlled metric evaluation.
* **Experiment Tracking:**
  * Future experiments and potential improvements will be systematically tracked via Git commits, branches, and version tags.
