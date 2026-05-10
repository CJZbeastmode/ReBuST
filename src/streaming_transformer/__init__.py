"""Streaming transformer package."""

from .model import StreamingMILTransformer
from .data import (
    WSIEmbeddingDataset,
    build_items_from_labels_json,
    build_items_from_pt_labels,
    build_items_from_pt_labels_with_map,
    collate_wsi_batch,
    split_items,
)
from .engine import train_one_epoch, evaluate

__all__ = [
    "StreamingMILTransformer",
    "WSIEmbeddingDataset",
    "build_items_from_labels_json",
    "build_items_from_pt_labels",
    "build_items_from_pt_labels_with_map",
    "collate_wsi_batch",
    "split_items",
    "train_one_epoch",
    "evaluate",
]
