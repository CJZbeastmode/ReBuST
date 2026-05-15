"""Step 8: train ablation patch selectors (SASHA, EvoPS, DualAttention)."""

import argparse
import importlib
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

METHODS = {
    "sasha": "src.ablation_patch_selector.SASHA.train_pipeline",
    "evops": "src.ablation_patch_selector.EvoPS.train_pipeline",
    "dualattention": "src.ablation_patch_selector.DualAttention.train_pipeline",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ablation patch selectors (SASHA, EvoPS, DualAttention)",
        add_help=True,
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=sorted(METHODS.keys()),
        help="Patch selector to train",
    )
    args, remainder = parser.parse_known_args()

    module = importlib.import_module(METHODS[args.method])

    sys.argv = [module.__file__] + remainder
    if hasattr(module, "main"):
        module.main()
        return
    if hasattr(module, "parse_args") and hasattr(module, "train_sasha"):
        module.train_sasha(module.parse_args())
        return
    if hasattr(module, "parse_args") and hasattr(module, "train_evops"):
        module.train_evops(module.parse_args())
        return
    if hasattr(module, "parse_args") and hasattr(module, "train_raza"):
        module.train_raza(module.parse_args())
        return
    raise RuntimeError(f"Unsupported training module: {module}")


if __name__ == "__main__":
    main()
