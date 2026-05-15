# MA

This repository is a research codebase for building, training, and evaluating
WSI patch-selection pipelines and downstream classifiers. The thesis contains
the scientific narrative; this README focuses on technical usage.

## Setup

This project expects Python 3.10+ and a working PyTorch install. Use a clean
environment and install the dependencies from requirements.txt.

```fish
# from repository root
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use conda, create an environment first and install the same requirements
into it. GPU acceleration is optional but recommended for training.

## Repository layout

- src/: Core modules and training/inference code
  - streaming_transformer/: ReBuST model (training + inference)
  - ablation_patch_selector/: Patch selector ablations (SASHA, EvoPS, DualAttention)
  - ablation_classifier/: Classifier ablations (aggregation transformer, mamba)
  - global_budget_enforcer/: HUMBE budget enforcement
  - training/: RL and supervised training code
  - inference/: RL, supervised, and greedy inference utilities
  - utils/: WSI handling, embeddings, patch scoring, helpers
- pipeline/: Step-by-step runnable entry points (end-to-end pipelines)
- scripts/: Standalone utilities and experiments
- data/: Metrics, visualizations, and intermediate artifacts
- docs/: Notes and supporting documentation

## Quick start

Most users should follow the pipeline steps documented in pipeline/README.md.

```fish
# from repository root
python pipeline/step1_create_wsi_objects.py
python pipeline/step2_global_budget_enforcer.py
python pipeline/step3_train_a2c_model.py
python pipeline/step4_infer_a2c_model.py
python pipeline/step5_extract_embeddings.py
python pipeline/step6_train_streaming_transformer.py
python pipeline/step7_infer_streaming_transformer.py
```

## Ablations

Patch selector ablations and classifier ablations are also wired in pipeline/.

```fish
python pipeline/step8_train_ablation_patch_selectors.py --method sasha
python pipeline/step9_infer_ablation_patch_selectors.py --method sasha

python pipeline/step10_train_ablation_classifiers.py --family aggregation --method CLAM
python pipeline/step11_infer_ablation_classifiers.py --family mamba
```

## Utilities

Scripts in scripts/ include data preparation, validation, and analysis helpers.
Each script has inline help and can be run directly.

Example:

```fish
python scripts/validate_info_gain_proxy.py \
  --csv data/benchmark/proxy_eval.csv \
  --score-cols entropy_score contrastive_score centroid_score \
  --gain-col downstream_gain \
  --out-csv data/benchmark/proxy_eval_ranked.csv \
  --out-json data/benchmark/proxy_eval_ranked.json \
  --plot-dir data/visualizations/proxy_eval
```

## Notes

- The codebase expects WSI files (.svs) and precomputed .pt embeddings in the
  folder layouts used by the pipeline steps.
- All training/inference entry points are in src/ and are wrapped by pipeline/.
- If you change dataset layout or file naming, update the corresponding pipeline
  step or dataset loader in src/.