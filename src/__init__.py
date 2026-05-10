"""Medical Image Analysis (MA) package.

Top-level modules:
  - ablation_classifier: Classifier ablation models and training
  - ablation_patch_selector: Patch selector ablations and training
  - downloader: Dataset acquisition utilities
  - downstream_task: Downstream evaluation tasks
  - global_budget_enforcer: Budget enforcement logic
  - inference: Inference and evaluation utilities
  - pipelines: End-to-end pipeline scripts
  - streaming_transformer: Streaming transformer models
  - supervised_data_collection: Supervised data collection utilities
  - training: Training entry points and helpers
  - utils: Shared utilities
"""

__version__ = "0.1.0"
__all__ = [
    "ablation_classifier",
    "ablation_patch_selector",
    "downloader",
    "downstream_task",
    "global_budget_enforcer",
    "inference",
    "pipelines",
    "streaming_transformer",
    "supervised_data_collection",
    "training",
    "utils",
]
