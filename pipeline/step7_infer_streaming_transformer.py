"""Step 7: run inference for the ReBuST streaming transformer head.

Thin wrapper over src/streaming_transformer/infer_streaming_mil.py.
"""

import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.streaming_transformer import infer_streaming_mil


def main() -> None:
    args = infer_streaming_mil.parse_args()
    infer_streaming_mil.main(args)


if __name__ == "__main__":
    main()
