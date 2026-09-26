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

---

# Kaggle Execution

The entire pipeline is automated via `run_kaggle.py` for headless execution in a Kaggle GPU notebook or Linux compute instance with **zero source-code editing**.

### 1. Minimal Execution Command

In a Kaggle notebook with GPU enabled (e.g. Tesla T4 or P100):

```bash
# Clone the repository and switch to the experiment branch
git clone https://github.com/CipherCraze/Amazon-ML-Final.git
cd Amazon-ML-Final
git checkout exp/kaggle-runner

# Install dependencies
pip install -r requirements.txt

# Run Experiment 002A (15 Baseline Features + Candidate Rank)
python run_kaggle.py \
    --experiment 002a \
    --data-root /kaggle/input/amazon-ml-challenge-2026/dataset \
    --output-root /kaggle/working/output
```

### 2. Pre-Flight Validation (Dry Run)

To verify all input paths, datasets, test candidates, and CUDA availability without initiating long-running jobs:

```bash
python run_kaggle.py --experiment 002a --dry-run
```

### 3. Execution Pipeline Details

- **Dataset Mounting (`--data-root`):** Point to the Kaggle input directory containing `train/` and `test/` TSV files. The runner automatically resolves source files whether nested under `train/` or in the root of the data folder.
- **Output Directory (`--output-root`):** Artifacts and submission files are created under `/kaggle/working/output` (or `./output` locally).
- **Automated Training Blocking:** `run_kaggle.py` automatically detects if `full_train_candidate_pairs.tsv` is absent and runs dense semantic GPU blocking ($k=100$) using `intfloat/multilingual-e5-small`. If CUDA is not available, it fails immediately to prevent slow CPU execution.
- **Reusing Test Candidates:** The existing, verified test candidate pool (`output/candidate_pairs.tsv`) is preserved and reused directly. Test blocking is **never regenerated**.
- **Automated Model Training & Threshold Calibration:** LightGBM trains on the 16-feature representation (`15 baseline features + candidate_rank`) and dynamically tunes the exact competition macro $F_{0.5}$ metric on the undownsampled validation split, discovering the optimal decision threshold and singleton cutoff without hardcoding.
- **Automated Inference & Submission Generation:** `matching_results.tsv` is generated directly from the trained model and test candidate pairs.
- **Automated Official Validation:** The runner automatically invokes `utils/validate_submission.py` against `matching_results.tsv` and `candidate_pairs.tsv`, printing the compliance report and verifying that the final output is 100% submission-ready.
- **Zero Code Editing:** All file paths, database connections, and thresholds are managed dynamically through command-line arguments and configuration files. No manual constant edits are required.

