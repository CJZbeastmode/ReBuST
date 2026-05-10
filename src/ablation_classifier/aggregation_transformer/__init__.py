"""Aggregation Transformer package (ABMIL/CLAM pooling + transformer)."""

from .model import AggregationTransformer
from .data import (
    WSIEmbeddingDataset,
    build_items_from_pt_labels,
    build_items_from_pt_labels_with_map,
    collate_wsi_batch,
)
from .engine import train_one_epoch, evaluate

__all__ = [
    "AggregationTransformer",
    "WSIEmbeddingDataset",
    "build_items_from_pt_labels",
    "build_items_from_pt_labels_with_map",
    "collate_wsi_batch",
    "train_one_epoch",
    "evaluate",
]
