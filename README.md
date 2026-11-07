# Multi-Filter Infrared Pattern Classification for HFC Refrigerant Leakage Detection

Code accompanying the manuscript *Multi-Filter Infrared Pattern Classification for HFC Refrigerant Leakage Detection Using CNN under Outdoor Conditions* (Tukadi, M. Rivai, and T. Mujiono, Institut Teknologi Sepuluh Nopember).

## Contents

| File | Purpose | Manuscript |
| --- | --- | --- |
| `notebooks/01_main_session_grouped_pipeline.ipynb` | Session-grouped five-fold cross-validation, region ablation and spatial controls, final fixed-epoch model, independent-session holdout | Sections 2.3–2.6, 3.2–3.4 |
| `notebooks/02_classifier_baselines.ipynb` | Fourteen comparator classifiers under the identical grouped protocol, session-level bootstrap | Section 3.7, Table 11 |
| `notebooks/03_no_filter_control.ipynb` | Physical no-filter control (development and holdout) | Section 2.7, Table 4 |
| `scripts/04_lamp_response_test.py` | Qualitative detector response to a broadband source through each optical element | Section 2.1, Fig. 3 |
| `scripts/05_model_size_latency.py` | Model size and single-frame latency | Section 3.7, Table 11 |
| `scripts/06_cnn_int8_holdout.py` | Holdout accuracy after int8 weight quantization | Section 3.7 |
| `scripts/07_no_filter_holdout_check.py` | Verification of the no-filter holdout archives | Response to reviewers |
| `scripts/08_reviewer_round2_analyses.py` | Random-forest region ablation, session-level uncertainty, early-stopping and augmentation-order controls, operating threshold | Sections 3.2–3.4 |

## Requirements

See `requirements.txt` (Python 3.11.8, TensorFlow 2.21.0, scikit-learn 1.8.0). All analyses run on CPU.

## Usage

1. Place the data folders (`X_learning5`, `X_testing5`, and the no-filter archives) on the local computer.
2. Edit the paths in the settings block at the top of each notebook or script.
3. Run the notebooks in order (01 → 03); the scripts in `scripts/` use the outputs of notebook 01 or 03.

Fold assignments are read from `source_fold_manifest.csv` produced by notebook 01, so all analyses use identical session-grouped partitions. Random seeds are fixed at 42.

## Data availability

The thermal frames, background references, fold manifests, and per-sample predictions are not included in this repository. They were provided to the reviewers and are available from the corresponding author upon reasonable request.

## Citation

If you use this code, please cite the article above (citation details will be added after publication).
