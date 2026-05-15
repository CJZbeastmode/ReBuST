"""Step 11: infer ablation classifiers (AggregationTransformer or Mamba MIL)."""

import argparse
import importlib
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

METHODS = {
    "aggregation": "src.ablation_classifier.aggregation_transformer.infer_aggregation_transformer",
    "mamba": "src.ablation_classifier.mamba.infer_mamba_mil",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer ablation classifiers (aggregation transformer or mamba).",
        add_help=True,
    )
    parser.add_argument(
        "--family",
        required=True,
        choices=sorted(METHODS.keys()),
        help="Classifier family to run",
    )
    args, remainder = parser.parse_known_args()

    module = importlib.import_module(METHODS[args.family])
    sys.argv = [module.__file__] + remainder

    if hasattr(module, "parse_args") and hasattr(module, "main"):
        module.main(module.parse_args())
        return
    raise RuntimeError(f"Unsupported inference module: {module}")


if __name__ == "__main__":
    main()
