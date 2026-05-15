"""Step 6: train the ReBuST streaming transformer head.

Thin wrapper over src/streaming_transformer/train_streaming_mil.py.
"""

import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.streaming_transformer import train_streaming_mil


def main() -> None:
    args = train_streaming_mil.parse_args()
    train_streaming_mil.train(args)


if __name__ == "__main__":
    main()
