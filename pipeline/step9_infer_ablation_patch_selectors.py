"""Step 9: infer ablation patch selectors (SASHA, EvoPS, DualAttention)."""

import argparse
import importlib
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

METHODS = {
    "sasha": "src.ablation_patch_selector.SASHA.infer_pipeline",
    "evops": "src.ablation_patch_selector.EvoPS.infer_pipeline",
    "dualattention": "src.ablation_patch_selector.DualAttention.infer_pipeline",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer ablation patch selectors (SASHA, EvoPS, DualAttention)",
        add_help=True,
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=sorted(METHODS.keys()),
        help="Patch selector to run",
    )
    args, remainder = parser.parse_known_args()

    module = importlib.import_module(METHODS[args.method])

    sys.argv = [module.__file__] + remainder
    if hasattr(module, "main"):
        module.main()
        return
    raise RuntimeError(f"Unsupported inference module: {module}")


if __name__ == "__main__":
    main()
