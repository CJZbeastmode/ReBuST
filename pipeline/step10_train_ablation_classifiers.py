"""Step 10: train ablation classifiers (AggregationTransformer or Mamba MIL)."""

import argparse
import importlib
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

METHODS = {
    "aggregation": "src.ablation_classifier.aggregation_transformer.train_aggregation_transformer",
    "mamba": "src.ablation_classifier.mamba.train_mamba_mil",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ablation classifiers (aggregation transformer or mamba).",
        add_help=True,
    )
    parser.add_argument(
        "--family",
        required=True,
        choices=sorted(METHODS.keys()),
        help="Classifier family to train",
    )
    args, remainder = parser.parse_known_args()

    module = importlib.import_module(METHODS[args.family])
    sys.argv = [module.__file__] + remainder

    if hasattr(module, "parse_args") and hasattr(module, "train"):
        module.train(module.parse_args())
        return
    if hasattr(module, "main"):
        module.main()
        return
    raise RuntimeError(f"Unsupported training module: {module}")


if __name__ == "__main__":
    main()
