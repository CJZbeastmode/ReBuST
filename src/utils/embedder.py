"""Module for embedder."""

import numpy as np
import os
import torch
import hashlib
import torch.nn.functional as F
import json
from transformers import CLIPModel, CLIPProcessor
from conch.open_clip_custom import tokenize, get_tokenizer, create_model_from_pretrained

DEFAULT_KEYWORD_PATH = "data/pathology_keywords/default.json"


class Embedder:
    """
    Embedder

    Unified abstraction for computing image embeddings and text–image similarity
    in computational pathology workflows.

    The class supports two backends:
        - PLIP  (CLIP-style pathology model)
        - CONCH (open-clip based pathology model)

    Design principles:
    ------------------
    1. Robustness:
       All public-facing methods are exception-safe. Failures return neutral
       values (zero vectors or zero scores) instead of propagating errors.

    2. Numerical stability:
       NaN / Inf values are explicitly removed. All embeddings are normalized
       to unit length where applicable.

    3. Reproducibility:
       Deterministic preprocessing and optional caching are used to ensure
       consistent behavior across repeated calls.

    This class is intended to be used inside long-running pipelines
    (e.g. RL environments, greedy inference, ablations) where crashes
    are unacceptable.
    """

    def __init__(
        self,
        img_backend="plip",
        keyword_path=DEFAULT_KEYWORD_PATH,
        device=None,
        model_path: str | None = None,
    ):
        """
        Initialize the embedding backend and auxiliary resources.

        Parameters
        ----------
        img_backend : str
            Which image encoder backend to use by default ('plip' or 'conch').
        keyword_path : str
            Path to a JSON file containing pathology-related keywords
            (category → list of terms).
        """
        self.img_backend = img_backend.lower()

        self.device = (
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        )

        # --------------------------------------------------
        # Model / processor initialization
        # --------------------------------------------------
        # This block initializes the chosen vision–language model.
        # Any failure here is considered fatal, as no embeddings
        # can be computed without a valid model.
        try:
            if img_backend.lower() == "conch":
                hf_token = os.environ.get("HF_AUTH_TOKEN")
                if not hf_token:
                    raise RuntimeError(
                        "HF_AUTH_TOKEN not set in environment. Please set it in your .env.secret file."
                    )
                self.model, self.processor = create_model_from_pretrained(
                    "conch_ViT-B-16", "hf_hub:MahmoodLab/conch", hf_auth_token=hf_token
                )
            elif img_backend.lower() == "plip":
                self.model = CLIPModel.from_pretrained("vinid/plip")
                self.processor = CLIPProcessor.from_pretrained("vinid/plip")
            else:
                raise ValueError
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize image backend '{img_backend}': {e}"
            )

        # --------------------------------------------------
        # Optional fine-tuned checkpoint load
        # --------------------------------------------------
        if model_path:
            try:
                payload = torch.load(model_path, map_location=self.device)
                state_dict = payload.get("model_state_dict", payload)
                missing, unexpected = self.model.load_state_dict(
                    state_dict, strict=False
                )
                if missing or unexpected:
                    print(
                        f"[Embedder] checkpoint loaded with missing={len(missing)} unexpected={len(unexpected)}"
                    )
            except Exception as exc:
                raise RuntimeError(f"Failed to load embedder checkpoint: {exc}")

        # --------------------------------------------------
        # Keyword loading (non-fatal)
        # --------------------------------------------------
        # Keywords are used for text–image similarity scoring.
        # Failure to load them must not crash the pipeline.
        self.keyword_path = keyword_path
        self.keyword_list = []
        try:
            with open(self.keyword_path, "r", encoding="utf-8") as fh:
                kw_map = json.load(fh)
                if isinstance(kw_map, dict):
                    for _, v in kw_map.items():
                        if isinstance(v, list):
                            self.keyword_list.extend([str(x) for x in v])
        except Exception:
            # Leave keyword list empty if loading fails
            self.keyword_list = []

        # --------------------------------------------------
        # Optional CONCH tokenizer
        # --------------------------------------------------
        # The tokenizer is optional and only required for CONCH-based
        # text encoding. Its absence should not affect PLIP workflows.
        try:
            self._tokenizer = get_tokenizer()
        except Exception:
            self._tokenizer = None

        # --------------------------------------------------
        # Internal caches
        # --------------------------------------------------
        # These caches reduce redundant computation during repeated
        # embedding and similarity calls.
        self._cached_text_embs = None
        self._cached_conch_text_embs = None
        self._cached_plip_text_embs = None
        self._img_cache = {}

    # --------------------------------------------------
    # Image embeddings (CONCH backend)
    # --------------------------------------------------
    def _conch_img_emb(self, patch):
        """
        Compute a CONCH image embedding.

        The returned embedding is:
        - 1D
        - L2-normalized
        - Located on CPU

        Any failure during preprocessing, model inference, or normalization
        results in a zero vector.
        """
        # Attempt to create a deterministic cache key from raw image bytes
        try:
            arr = np.array(patch)
            key = hashlib.sha1(arr.tobytes()).hexdigest()
        except Exception:
            key = None

        if key is not None and key in self._img_cache:
            return self._img_cache[key]

        # Image preprocessing (robust to processor API differences)
        try:
            try:
                image_tensor = self.processor(patch).unsqueeze(0)
            except Exception:
                image_tensor = self.processor(
                    images=np.array(patch), return_tensors="pt"
                )["pixel_values"]
        except Exception:
            return torch.zeros(512)

        # Forward pass through the model
        try:
            with torch.inference_mode():
                if hasattr(self.model, "encode_image"):
                    image_emb = self.model.encode_image(
                        image_tensor, proj_contrast=False, normalize=False
                    )
                elif hasattr(self.model, "get_image_features"):
                    image_emb = self.model.get_image_features(images=image_tensor)
                else:
                    return torch.zeros(512)
        except Exception:
            return torch.zeros(512)

        # Post-processing and normalization
        emb = image_emb.detach().cpu()
        if emb.ndim == 2 and emb.shape[0] == 1:
            emb = emb.squeeze(0)

        emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        n = emb.norm(p=2)
        if not torch.isfinite(n) or n < 1e-6:
            emb = torch.zeros_like(emb)
        else:
            emb = emb / (n + 1e-12)

        if key is not None:
            self._img_cache[key] = emb

        return emb

    # --------------------------------------------------
    # Image embeddings (PLIP backend)
    # --------------------------------------------------
    def _plip_img_emb(self, patch):
        """
        Compute a PLIP image embedding.

        Additional safeguards are applied:
        - Detection of near-constant (blank) patches
        - Explicit NaN / Inf removal
        - Stable normalization
        """
        try:
            arr = np.array(patch, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        except Exception:
            return torch.zeros(512)

        # Early exit for visually empty patches
        try:
            if arr.std() < 1e-3:
                return torch.zeros(512)
        except Exception:
            pass

        # Model inference
        try:
            inputs = self.processor(images=arr, return_tensors="pt")
            with torch.no_grad():
                if hasattr(self.model, "get_image_features"):
                    feat = self.model.get_image_features(**inputs).squeeze(0)
                elif hasattr(self.model, "encode_image"):
                    img = inputs.get("pixel_values") or inputs.get("images")
                    feat = self.model.encode_image(img).squeeze(0)
                else:
                    return torch.zeros(512)
        except Exception:
            return torch.zeros(512)

        # Normalization and numerical cleanup
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        norm = feat.norm(p=2)
        if not torch.isfinite(norm) or norm < 1e-6:
            return torch.zeros_like(feat)

        return (feat / (norm + 1e-12)).cpu()

    # --------------------------------------------------
    # Text similarity (CONCH)
    # --------------------------------------------------
    def _conch_text_sim(self, patch, keywords=None, aggregate="max"):
        """
        Compute text–image similarity using the CONCH backend.

        Similarity is computed as a cosine similarity between the image
        embedding and a set of text embeddings.

        Aggregation strategies:
            - 'max'  : strongest semantic match
            - 'mean' : average semantic relevance
        """
        try:
            texts = keywords if keywords is not None else self.keyword_list
            if not texts:
                return 0.0

            # Cache text embeddings for repeated default usage
            if keywords is None and self._cached_conch_text_embs is not None:
                text_embs = self._cached_conch_text_embs
            else:
                text_tokens = tokenize(self._tokenizer, texts)
                with torch.inference_mode():
                    text_embs = self.model.encode_text(text_tokens, normalize=True)
                if keywords is None:
                    self._cached_conch_text_embs = text_embs

            image = self.processor(patch).unsqueeze(0)
            with torch.inference_mode():
                image_emb = self.model.encode_image(
                    image, proj_contrast=True, normalize=True
                )

            sim_scores = (image_emb @ text_embs.T).squeeze(0)

            return float(sim_scores.max() if aggregate == "max" else sim_scores.mean())
        except Exception:
            return 0.0

    # --------------------------------------------------
    # Text similarity (PLIP)
    # --------------------------------------------------
    def _plip_text_sim(self, emb, top_k=3):
        """
        Category-aware PLIP text similarity.

        For each pathology category, multiple prompt variants are used.
        The final score corresponds to the strongest category evidence.

        The output is discretized to an integer range [0, 100] for
        stability and downstream compatibility.
        """
        try:
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, dtype=torch.float32)

            emb = emb / (emb.norm() + 1e-12)
            emb = emb.unsqueeze(0)
        except Exception:
            return 0

        try:
            if not hasattr(self, "_plip_text_by_cat"):
                self._plip_text_by_cat = {}
                with open(self.keyword_path, "r", encoding="utf-8") as fh:
                    kw_map = json.load(fh)

                # Precompute text embeddings per category
                for cat, keywords in kw_map.items():
                    prompts = []
                    for kw in keywords:
                        prompts.extend(
                            [
                                f"H&E histology patch showing {kw}",
                                f"pathology tissue with {kw}",
                                f"microscopy image of {kw}",
                            ]
                        )

                    text_inputs = self.processor(
                        text=prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )

                    with torch.no_grad():
                        feats = self.model.get_text_features(**text_inputs)

                    feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
                    self._plip_text_by_cat[cat] = feats.cpu()

            category_scores = []
            for feats in self._plip_text_by_cat.values():
                sims = F.cosine_similarity(emb, feats.to(emb.dtype), dim=-1)
                k = min(top_k, sims.numel())
                category_scores.append(torch.topk(sims, k)[0].mean().item())

            best_score = max(category_scores)

            # Map cosine similarity range to a stable integer scale
            return int(np.clip((best_score - 0.15) / 0.3 * 100, 0, 100))
        except Exception:
            return 0

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def img_emb(self, patch):
        """
        Compute an image embedding using the configured backend.

        This method is safe to call inside training loops and will
        never raise exceptions.
        """
        try:
            if self.img_backend == "conch":
                return self._conch_img_emb(patch)
            return self._plip_img_emb(patch)
        except Exception:
            return torch.zeros(512)

    def text_sim(
        self, patch_or_emb=None, keywords=None, aggregate="mean", backend=None, k=3
    ):
        """
        Unified text–image similarity interface.

        Accepts either:
            - a raw image patch
            - a precomputed embedding tensor

        Always returns a valid scalar value, even in the presence of
        model or preprocessing failures.
        """
        try:
            be = (backend or self.img_backend or "plip").lower()
            if be == "conch":
                return float(
                    self._conch_text_sim(
                        patch_or_emb, keywords=keywords, aggregate=aggregate
                    )
                )

            emb = (
                patch_or_emb
                if isinstance(patch_or_emb, torch.Tensor)
                else self._plip_img_emb(patch_or_emb)
            )
            return int(self._plip_text_sim(emb))
        except Exception:
            return 0
