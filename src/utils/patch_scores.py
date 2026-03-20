from torch.nn.functional import cosine_similarity
from .embedder import Embedder
import numpy as np
import cv2
import json
import os
import torch


class PatchScoreModule:
    """
    Abstract base class for all patch scoring modules.

    This interface defines the minimal contract required by the
    DynamicPatchEnv and downstream RL agents.

    Design assumptions:
    -------------------
    - All methods must be side-effect free.
    - No method may raise exceptions during training or inference.
    - Any failure must degrade gracefully to a neutral score (typically 0.0).
    - Scores are assumed to be comparable within a module, but not necessarily
      across different modules.

    Each concrete subclass encodes a different inductive bias about what
    constitutes a “good” zoom decision in a WSI.
    """

    def compute_stop(self, **kwargs):
        """Score associated with terminating exploration at the current patch."""
        raise NotImplementedError

    def compute_zoom(self, **kwargs):
        """Score associated with zooming into child patches."""
        raise NotImplementedError

    def compute_diff(self, **kwargs):
        """
        Convenience method computing the relative preference for ZOOM vs STOP.
        Typically implemented as: s_zoom − s_stop.
        """
        raise NotImplementedError

    def infer(self, **kwargs):
        """
        Deterministic inference rule mapping scores to an action.
        Used for greedy or evaluation-time policies.
        """
        raise NotImplementedError

    def rl_parameters(self, **kwargs):
        """
        Optional module-specific RL hyperparameters.
        These values may be used to tune exploration, penalties, or learning rate.
        """
        return NotImplementedError


# ==========================================================
# Image similarity score
# ==========================================================
class ImgSimScore(PatchScoreModule):
    """
    Image-similarity–based score.

    Motivation:
    -----------
    Zooming is rewarded when child patches are *visually dissimilar*
    from the parent patch, encouraging exploration of new visual content.

    This score is agnostic to semantics and relies purely on embedding
    geometry, making it a strong unsupervised baseline.
    """

    def __init__(self, weight=10.0, embedder=None, agg="mean", **kwargs):
        if agg not in ["mean", "max"]:
            raise ValueError(f"Invalid aggregation method: {agg}")

        self.weight = weight  # Global scaling factor for reward magnitude
        self.embedder = embedder  # Shared image embedding backend (PLIP / CONCH)
        self.agg = agg  # Aggregation method for multiple children

    def compute_stop(self, **kwargs):
        """
        STOP has no intrinsic reward in this module.
        All preference is expressed via the zoom score.
        """
        return 1.0  # Only meaningful structural difference would result in zooming

    def compute_zoom(self, parent_patch=None, child_patches=None, **kwargs):
        """
        Compute zoom reward as inverse similarity between parent and children.

        Similarity is measured via cosine similarity in embedding space.
        The reward is proportional to (1 − similarity), i.e. information gain.
        """
        if parent_patch is None or not child_patches or self.embedder is None:
            return 0.0

        try:
            ep = self.embedder.img_emb(parent_patch)
        except Exception:
            return 0.0

        sims = []
        for p in child_patches:
            try:
                ec = self.embedder.img_emb(p)
                sim_t = cosine_similarity(ep, ec, dim=0)
                sim_val = (
                    float(sim_t.mean().item())
                    if sim_t.numel() > 1
                    else float(sim_t.item())
                )
                sims.append(sim_val)
            except Exception:
                continue

        if len(sims) == 0:
            return 0.0

        try:
            if self.agg == "mean":
                return self.weight * (1.0 - float(np.mean(sims)))
            elif self.agg == "max":
                # Conservative variant: penalize the most similar child
                return self.weight * (1.0 - float(np.min(sims)))
        except Exception:
            return 0.0

        return 0.0

    def compute_diff(self, parent_patch=None, child_patches=None, **kwargs):
        """
        Relative preference for zooming vs stopping.
        """
        try:
            s_stop = self.compute_stop(parent_patch=parent_patch, **kwargs)
            s_zoom = self.compute_zoom(
                parent_patch=parent_patch,
                child_patches=child_patches,
                agg=self.agg,
                **kwargs,
            )
            return s_zoom - s_stop
        except Exception:
            return 0.0

    def infer(self, s_stop, s_zoom):
        """
        Greedy inference rule:
        - If both scores are non-positive, STOP.
        - Otherwise choose the higher score.
        """
        try:
            if s_stop <= 0 and s_zoom <= 0:
                return 0
            return 1 if (s_zoom >= s_stop) else 0
        except Exception:
            return 0

    def rl_parameters(self):
        """
        Recommended RL hyperparameters for this reward structure.

        The values reflect:
        - Dense, smooth rewards
        - Short-horizon exploration
        - Mild entropy regularization
        """
        return {
            "ZOOM_COST": 0.3,
            "DEPTH_COST": 0.05,
            "MAX_ZOOM_FRAC": 0.5,
            "OVERZOOM_PENALTY": 0.0,
            "ENTROPY_BETA": 0.02,
            "GAMMA": 0.95,
            "LR": 1e-4,
        }


# ==========================================================
# Text–image alignment score
# ==========================================================
class TextAlignScore(PatchScoreModule):
    """
    Semantic alignment score between image patches and pathology text prompts.

    Motivation:
    -----------
    Zooming is encouraged when child patches increase semantic alignment
    with domain-specific pathology concepts.

    This module introduces weak supervision via language.
    """

    def __init__(self, weight=1.0, embedder=None, k=3, agg="mean", **kwargs):
        if agg not in ["mean", "max"]:
            raise ValueError(f"Invalid aggregation method: {agg}")

        self.weight = weight
        self.embedder = Embedder(img_backend="plip") if embedder is None else embedder  # Shared vision–language model
        self.k = k  # Top-k aggregation (currently implicit)
        self.agg = agg

    def compute_stop(self, parent_patch=None, **kwargs):
        """
        STOP score reflects how well the current patch aligns with pathology text.
        """
        if parent_patch is None or self.embedder is None:
            return 0.0
        try:
            s = self.embedder.text_sim(parent_patch)
            return self.weight * float(s)
        except Exception:
            return 0.0

    def compute_zoom(self, parent_patch=None, child_patches=None, **kwargs):
        """
        Zoom score aggregates semantic alignment across child patches.
        """
        if not child_patches or self.embedder is None:
            return 0.0

        scores = []
        for p in child_patches:
            try:
                scores.append(self.embedder.text_sim(p, aggregate=self.agg))
            except Exception:
                continue

        if len(scores) == 0:
            return 0.0

        try:
            if self.agg == "mean":
                return self.weight * float(np.mean(scores))
            elif self.agg == "max":
                return self.weight * float(np.max(scores))
        except Exception:
            return 0.0

        return 0.0

    def compute_diff(self, parent_patch=None, child_patches=None, **kwargs):
        """
        Relative semantic gain from zooming.
        """
        try:
            s_stop = self.compute_stop(parent_patch=parent_patch, **kwargs)
            s_zoom = self.compute_zoom(
                parent_patch=parent_patch,
                child_patches=child_patches,
                agg=self.agg,
                **kwargs,
            )
            return s_zoom - s_stop
        except Exception:
            return 0.0

    def infer(self, s_stop, s_zoom):
        """Greedy semantic decision rule."""
        try:
            return 1 if (s_zoom >= s_stop) else 0
        except Exception:
            return 0

    def rl_parameters(self):
        """
        RL hyperparameters tuned for sparse, higher-variance semantic rewards.
        """
        return {
            "ZOOM_COST": 2.0,
            "DEPTH_COST": 0.1,
            "MAX_ZOOM_FRAC": 0.5,
            "OVERZOOM_PENALTY": 0.0,
            "ENTROPY_BETA": 0.01,
            "GAMMA": 0.95,
            "LR": 1e-4,
        }


# ==========================================================
# Tissue presence (reward-only)
# ==========================================================
class TissuePresenceScore(PatchScoreModule):
    """
    Binary tissue presence reward.

    Motivation:
    -----------
    Encourages exploration of regions containing tissue while remaining
    neutral toward blank/background regions.

    This module provides a very simple heuristic baseline.
    """

    def __init__(self, weight=1.0, blank_thr=230, agg="any", **kwargs):
        if agg not in ["any", "all"]:
            raise ValueError(f"Invalid aggregation method: {agg}")

        self.weight = weight
        self.blank_thr = blank_thr  # Mean intensity threshold for blank detection
        self.agg = agg

    def _is_blank(self, patch):
        """Heuristic blank detection via mean pixel intensity."""
        try:
            return np.array(patch).mean() > self.blank_thr
        except Exception:
            return True

    def compute_stop(self, parent_patch=None, **kwargs):
        """Reward stopping if the current patch contains tissue."""
        if parent_patch is None:
            return 0.0
        try:
            return self.weight if not self._is_blank(parent_patch) else 0.0
        except Exception:
            return 0.0

    def compute_zoom(self, parent_patch=None, child_patches=None, **kwargs):
        """
        Reward zooming if tissue is detected among child patches.
        """
        if not child_patches:
            return 0.0

        try:
            if self.agg == "any":
                return (
                    self.weight
                    if any(not self._is_blank(p) for p in child_patches)
                    else 0.0
                )
            elif self.agg == "all":
                return (
                    self.weight
                    if all(not self._is_blank(p) for p in child_patches)
                    else 0.0
                )
        except Exception:
            return 0.0

        return 0.0

    def compute_diff(self, parent_patch=None, child_patches=None, **kwargs):
        """Relative tissue gain from zooming."""
        try:
            return self.compute_zoom(
                parent_patch=parent_patch,
                child_patches=child_patches,
                agg=self.agg,
                **kwargs,
            ) - self.compute_stop(parent_patch=parent_patch, **kwargs)
        except Exception:
            return 0.0

    def infer(self, s_stop, s_zoom):
        """Prefer zoom if it increases tissue presence."""
        try:
            return 1 if (s_zoom > s_stop) else 0
        except Exception:
            return 0


# ==========================================================
# Tissue presence (explicit penalty version)
# ==========================================================
class TissuePresencePenalty(PatchScoreModule):
    """
    Signed tissue reward with explicit penalties.

    Motivation:
    -----------
    Explicitly penalizes blank/background regions instead of merely
    withholding reward. This produces sharper gradients and stronger
    exploration pressure.

    This module is *not* equivalent to TissuePresenceScore and should
    be treated as a distinct experimental condition.
    """

    def __init__(self, weight=1.0, blank_thr=230, agg="any", **kwargs):
        if agg not in ["any", "all"]:
            raise ValueError(f"Invalid aggregation method: {agg}")
        self.weight = weight
        self.blank_thr = blank_thr
        self.agg = agg

    def _is_blank(self, patch):
        try:
            return np.array(patch).mean() > self.blank_thr
        except Exception:
            return True

    def compute_stop(self, parent_patch=None, **kwargs):
        """Reward tissue, penalize blank at the current level."""
        if parent_patch is None:
            return 0.0
        try:
            return self.weight if not self._is_blank(parent_patch) else -self.weight
        except Exception:
            return 0.0

    def compute_zoom(self, parent_patch=None, child_patches=None, **kwargs):
        """Signed reward based on tissue presence in child patches."""
        if not child_patches:
            return 0.0

        try:
            if self.agg == "any":
                return (
                    self.weight
                    if any(not self._is_blank(p) for p in child_patches)
                    else -self.weight
                )
            elif self.agg == "all":
                return (
                    self.weight
                    if all(not self._is_blank(p) for p in child_patches)
                    else -self.weight
                )
        except Exception:
            return 0.0

        return 0.0

    def compute_diff(self, parent_patch=None, child_patches=None, **kwargs):
        """Relative signed advantage of zooming."""
        try:
            return self.compute_zoom(
                parent_patch=parent_patch,
                child_patches=child_patches,
                agg=self.agg,
                **kwargs,
            ) - self.compute_stop(parent_patch=parent_patch, **kwargs)
        except Exception:
            return 0.0

    def infer(self, s_stop, s_zoom):
        """Zoom if it reduces blankness relative to stopping."""
        try:
            return 1 if (s_zoom > s_stop) else 0
        except Exception:
            return 0

    def rl_parameters(self):
        """
        Hyperparameters reflecting discrete, high-contrast rewards.
        """
        return {
            "ZOOM_COST": 0.2,
            "DEPTH_COST": 0.05,
            "MAX_ZOOM_FRAC": 0.5,
            "OVERZOOM_PENALTY": 0.0,
            "ENTROPY_BETA": 0.03,
            "GAMMA": 0.9,
            "LR": 1e-4,
        }


# ==========================================================
# Entropy-based score
# ==========================================================
class EntropyScore(PatchScoreModule):
    """
    Information-theoretic score based on relative grayscale entropy gain.

    Zooming is encouraged only if child patches exhibit a
    meaningful relative increase in entropy compared to the parent.
    """

    def __init__(self, weight=1.0, agg="max", tau=0.01, **kwargs):
        if agg not in ["mean", "max"]:
            raise ValueError(f"Invalid aggregation method: {agg}")
        self.weight = weight
        self.agg = agg
        self.tau = tau  # relative entropy gain threshold

    def _entropy(self, img_np):
        """Compute Shannon entropy of grayscale intensity distribution."""
        if img_np is None:
            return 0.0

        try:
            if img_np.ndim == 2:
                gray = img_np
            else:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_sum = hist.sum()
            if hist_sum <= 0:
                return 0.0

            hist = hist / hist_sum
            hist = np.maximum(hist, 1e-12)
            return float(-(hist * np.log(hist)).sum())
        except Exception:
            return 0.0

    def compute_stop(self, parent_patch=None, **kwargs):
        if parent_patch is None:
            return 0.0
        return self.weight * self._entropy(np.array(parent_patch))

    def compute_zoom(self, parent_patch=None, child_patches=None, **kwargs):
        if not child_patches:
            return 0.0

        entropies = []
        for cp in child_patches:
            e = self._entropy(np.array(cp))
            if e > 0:
                entropies.append(e)

        if len(entropies) == 0:
            return 0.0

        if self.agg == "mean":
            return self.weight * float(np.mean(entropies))
        elif self.agg == "max":
            return self.weight * float(np.max(entropies))

        return 0.0

    def compute_diff(self, parent_patch=None, child_patches=None, **kwargs):
        s_stop = self.compute_stop(parent_patch=parent_patch)
        if s_stop <= 0:
            return 0.0

        s_zoom = self.compute_zoom(
            parent_patch=parent_patch,
            child_patches=child_patches,
        )

        # Relative entropy gain
        return (s_zoom - s_stop) / (s_stop + 1e-6)

    def infer(self, s_stop, s_zoom):
        if s_stop <= 0:
            return 0
        diff = (s_zoom - s_stop) / (s_stop + 1e-6)
        return 1 if diff >= self.tau else 0


# ==========================================================
# Cancer-type centroid reward
# ==========================================================

class CancerTypeCentroidScore(PatchScoreModule):
    """
    PLIP-text-centroid reward for cancer-type-specific patch selection.

    Motivation:
    -----------
    Rather than aligning patches to generic pathology text, this module
    pre-embeds all morphologically discriminative phrases for a specific
    cancer type (loaded from data/pathology_keywords/<cancer_type>.json)
    and averages them into a single unit-norm centroid vector.

    At inference time the stop/zoom scores are the cosine similarity
    between the patch PLIP image embedding and that centroid, scaled
    by `weight`.  Because the centroid is computed once at construction,
    there is no per-patch overhead beyond a single PLIP image forward pass.

    Usage:
    ------
        scorer = CancerTypeCentroidScore(cancer_type="LUAD")
        s_stop = scorer.compute_stop(parent_patch)
        s_zoom = scorer.compute_zoom(child_patches=children)
    """

    KEYWORD_DIR = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "pathology_keywords")
    )

    PROMPT_TEMPLATES = [
        "H&E histology patch showing {kw}",
        "pathology tissue with {kw}",
        "microscopy image of {kw}",
    ]

    def __init__(self, cancer_type: str, weight: float = 1.0, embedder=None, agg: str = "mean", **kwargs):
        if agg not in ("mean", "max"):
            raise ValueError(f"CancerTypeCentroidScore: invalid aggregation '{agg}', choose 'mean' or 'max'.")
        self.cancer_type = cancer_type
        self.weight = weight
        self.agg = agg
        self.embedder = Embedder(img_backend="plip") if embedder is None else embedder
        self._centroid: torch.Tensor | None = None  # lazy-built on first use

    # ------------------------------------------------------------------
    # Centroid construction
    # ------------------------------------------------------------------

    def _build_centroid(self) -> torch.Tensor:
        """Load keyword JSON, encode all prompts with PLIP, return mean unit vector."""
        kw_path = os.path.join(self.KEYWORD_DIR, f"{self.cancer_type}.json")
        if not os.path.isfile(kw_path):
            raise FileNotFoundError(
                f"CancerTypeCentroidScore: no keyword file for '{self.cancer_type}' at {kw_path}. "
                f"Create data/pathology_keywords/{self.cancer_type}.json first."
            )

        with open(kw_path, "r", encoding="utf-8") as fh:
            kw_map = json.load(fh)

        all_prompts = [
            tmpl.format(kw=kw)
            for keywords in kw_map.values()
            for kw in keywords
            for tmpl in self.PROMPT_TEMPLATES
        ]

        text_inputs = self.embedder.processor(
            text=all_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        with torch.no_grad():
            feats = self.embedder.model.get_text_features(**text_inputs)

        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
        centroid = feats.mean(dim=0)
        centroid = centroid / (centroid.norm() + 1e-12)
        return centroid.cpu()

    def _get_centroid(self) -> torch.Tensor:
        if self._centroid is None:
            self._centroid = self._build_centroid()
        return self._centroid

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cos_sim(self, patch) -> float:
        """Cosine similarity between a patch (or tensor) and the centroid."""
        try:
            emb = (
                patch
                if isinstance(patch, torch.Tensor)
                else self.embedder._plip_img_emb(patch)
            )
            emb = emb.float()
            emb = emb / (emb.norm() + 1e-12)
            c = self._get_centroid().to(emb.dtype)
            return float(cosine_similarity(emb.unsqueeze(0), c.unsqueeze(0)).item())
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # PatchScoreModule interface
    # ------------------------------------------------------------------

    def compute_stop(self, parent_patch=None, **kwargs):
        """
        STOP score: cosine similarity of the current patch to the cancer-type centroid.
        """
        if parent_patch is None:
            return 0.0
        return self.weight * self._cos_sim(parent_patch)

    def compute_zoom(self, parent_patch=None, child_patches=None, **kwargs):
        """
        ZOOM score: aggregated cosine similarity of child patches to the centroid.
        """
        if not child_patches:
            return 0.0

        scores = []
        for p in child_patches:
            try:
                scores.append(self._cos_sim(p))
            except Exception:
                continue

        if not scores:
            return 0.0

        agg_score = float(np.mean(scores)) if self.agg == "mean" else float(np.max(scores))
        return self.weight * agg_score

    def compute_diff(self, parent_patch=None, child_patches=None, **kwargs):
        """
        Relative centroid-alignment gain from zooming.
        """
        try:
            s_stop = self.compute_stop(parent_patch=parent_patch)
            s_zoom = self.compute_zoom(child_patches=child_patches)
            return s_zoom - s_stop
        except Exception:
            return 0.0

    def infer(self, s_stop, s_zoom):
        """Zoom if the centroid alignment improves after zooming."""
        return 1 if s_zoom > s_stop else 0


# ==========================================================
# Biologically motivated hard-negative pairs
# ==========================================================

#: Maps each cancer type to its known hard-negative types.
#: These are pairs where morphological similarity is highest,
#: making discrimination diagnostically meaningful.
#:
#: Used by ContrastiveTextScore when neg_cancer_types="pairs".
#: If a cancer type has no entry here, "pairs" falls back to "all".
CONTRASTIVE_PAIRS: dict[str, list[str]] = {
    # Lung subtypes: adenocarcinoma vs squamous-cell (same organ, different cell type)
    # LUAD also confused with MESO (both can show papillary/glandular patterns)
    "LUAD": ["LUSC", "MESO"],
    "LUSC": ["LUAD"],
    # Colorectal: colon vs rectal (adjacent segments, nearly identical architecture)
    "COAD": ["READ", "STAD"],
    "READ": ["COAD", "STAD"],
    # Upper GI: stomach vs oesophagus (gastro-oesophageal junction ambiguity)
    "STAD": ["ESCA", "COAD"],
    "ESCA": ["STAD"],
    # Hepatobiliary: liver vs bile duct vs pancreas
    # LIHC and CHOL share the hepatic microenvironment; PAAD has similar ductal glands
    "LIHC": ["CHOL", "PAAD"],
    "PAAD": ["CHOL", "LIHC"],
    # Mesothelioma: pleural/peritoneal epithelioid growth confused with lung adenocarcinoma
    "MESO": ["LUAD", "PAAD"],
    # Biliary: cholangiocarcinoma vs hepatocellular vs pancreatic (glandular/ductal overlap)
    "CHOL": ["LIHC", "PAAD"],
    # Melanocytic tumours: cutaneous vs uveal melanoma (same cell lineage, different morphology)
    "SKCM": ["UVM"],
    "UVM":  ["SKCM"],
    # All other types are considered mutually confusable for the purposes of "all" negatives
    "all": ["LUAD", "LUSC", "COAD", "READ", "STAD", "ESCA", "LIHC", "PAAD", "MESO", "CHOL", "SKCM", "UVM"],
}


# ==========================================================
# Contrastive cancer-type centroid reward
# ==========================================================

class ContrastiveTextScore(PatchScoreModule):
    """
    Contrastive PLIP-text reward: class-specific alignment minus hardest negative.

    Motivation:
    -----------
    ``CancerTypeCentroidScore`` measures absolute alignment with a single
    cancer type's concept centroid.  A patch depicting generic necrosis or
    fibrosis can score high despite being diagnostically non-specific.

    This module adds a hard-negative margin term::

        r = cos(e_patch, c_pos) - max_j cos(e_patch, c_j)

    where {c_j} are centroids for a set of confusable cancer types.  The
    reward is positive only if the patch encodes features that are
    *discriminative for the correct class*, not merely "pathology-like".

    The hardest negatives are pre-selected by biology:
      - LUAD ↔ LUSC  (same organ, different morphology)
      - COAD ↔ READ  (adjacent bowel segments, nearly identical architecture)
      - STAD ↔ ESCA  (gastro-oesophageal junction ambiguity)

    neg_cancer_types accepts three forms:

      "all"            Every keyword file except pos_cancer_type (default).
      "pairs"          Look up CONTRASTIVE_PAIRS[pos_cancer_type] for the
                       biologically motivated hard negatives; falls back to
                       "all" when no entry exists for that cancer type.
      list[str]        Explicit list, e.g. ["LUSC"] or ["LUSC", "COAD"].

    Ablation axis (from easy to hardest):
      easy      → neg_cancer_types=["LIHC"]          (different organ, trivially distinct)
      hard      → neg_cancer_types=["LUSC"]           (same organ, different cell type)
      pairs     → neg_cancer_types="pairs"            (biologically curated CONTRASTIVE_PAIRS)
      exhaustive→ neg_cancer_types="all"              (every other keyword file)

    Centroids are built once at construction via ``CancerTypeCentroidScore``
    internals — no per-patch text encoder calls.

    Usage:
    ------
        # Biologically curated hard negatives (recommended)
        scorer = ContrastiveTextScore(
            pos_cancer_type="LUAD",
            neg_cancer_types="pairs",   # resolves to ["LUSC"]
        )
        # All other cancer types (default)
        scorer = ContrastiveTextScore(
            pos_cancer_type="LUAD",
            neg_cancer_types="all",
        )
        # Explicit list
        scorer = ContrastiveTextScore(
            pos_cancer_type="LUAD",
            neg_cancer_types=["LUSC", "COAD"],
        )
        r_stop = scorer.compute_stop(parent_patch)
        r_zoom = scorer.compute_zoom(child_patches=children)
    """

    @staticmethod
    def _resolve_neg_types(pos_cancer_type: str, neg_cancer_types) -> list:
        """
        Resolve neg_cancer_types to a concrete list of cancer-type strings.

        Modes
        -----
        "all"   →  every keyword JSON except pos_cancer_type and 'default'.
        "pairs" →  CONTRASTIVE_PAIRS[pos_cancer_type] if the entry exists and
                   all listed types have keyword files; otherwise falls back
                   to 'all' (with a warning printed to stdout).
        list    →  used as-is (caller's responsibility for correctness).
        """
        kw_dir = CancerTypeCentroidScore.KEYWORD_DIR
        # Enumerate all available cancer types once
        available = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(kw_dir)
            if f.endswith(".json")
            and os.path.splitext(f)[0] not in (pos_cancer_type, "default")
        )

        if neg_cancer_types == "all":
            print("[INFO] ContrastiveTextScore: using all available negative cancer types")
            resolved = CONTRASTIVE_PAIRS.get("all", available)

        elif neg_cancer_types == "pairs":
            pairs = CONTRASTIVE_PAIRS.get(pos_cancer_type)
            if pairs:
                # Keep only those that actually have keyword files
                resolved = [ct for ct in pairs if ct in available]
                if not resolved:
                    print(
                        f"[WARN] ContrastiveTextScore: no valid keyword files found for "
                        f"CONTRASTIVE_PAIRS['{pos_cancer_type}']={pairs}; "
                        f"falling back to neg_cancer_types='all'."
                    )
                    resolved = CONTRASTIVE_PAIRS.get("all", available)
                else:
                    print(
                        f"[INFO] ContrastiveTextScore: using curated hard negatives for "
                        f"'{pos_cancer_type}': {resolved}"
                    )
            else:
                print(
                    f"[INFO] ContrastiveTextScore: no CONTRASTIVE_PAIRS entry for "
                    f"'{pos_cancer_type}'; falling back to neg_cancer_types='all'."
                )
                resolved = CONTRASTIVE_PAIRS.get("all", available)

        else:
            resolved = list(neg_cancer_types)

        if not resolved:
            raise ValueError(
                f"ContrastiveTextScore: no negative cancer types found "
                f"(pos='{pos_cancer_type}', neg_cancer_types={neg_cancer_types!r})."
            )
        return resolved

    def __init__(
        self,
        pos_cancer_type: str,
        neg_cancer_types: str | list = "all",  # "all" | "pairs" | list[str]
        weight: float = 1.0,
        embedder=None,
        agg: str = "mean",
        **kwargs,
    ):
        if agg not in ("mean", "max"):
            raise ValueError(f"ContrastiveTextScore: invalid agg '{agg}', choose 'mean' or 'max'.")

        self.pos_cancer_type = pos_cancer_type
        self.neg_cancer_types = self._resolve_neg_types(pos_cancer_type, neg_cancer_types)
        self.weight = weight
        self.agg = agg
        # Shared embedder — all centroid scorers below reuse the same model instance
        self.embedder = Embedder(img_backend="plip") if embedder is None else embedder

        # Delegate centroid building to CancerTypeCentroidScore (lazy, cached internally)
        self._pos_scorer = CancerTypeCentroidScore(
            cancer_type=pos_cancer_type, weight=1.0, embedder=self.embedder
        )
        self._neg_scorers = [
            CancerTypeCentroidScore(cancer_type=ct, weight=1.0, embedder=self.embedder)
            for ct in self.neg_cancer_types
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _contrastive_score(self, patch) -> float:
        """
        cos(e, c_pos) - max_j cos(e, c_neg_j)

        Can be negative when the patch looks more like a negative class.
        """
        try:
            emb = (
                patch
                if isinstance(patch, torch.Tensor)
                else self.embedder._plip_img_emb(patch)
            )
            emb = emb.float()
            emb = emb / (emb.norm() + 1e-12)

            pos_c = self._pos_scorer._get_centroid().to(emb.dtype)
            pos_sim = float(cosine_similarity(emb.unsqueeze(0), pos_c.unsqueeze(0)).item())

            neg_sims = []
            for neg_scorer in self._neg_scorers:
                neg_c = neg_scorer._get_centroid().to(emb.dtype)
                neg_sims.append(
                    float(cosine_similarity(emb.unsqueeze(0), neg_c.unsqueeze(0)).item())
                )

            hardest_neg = max(neg_sims)
            return pos_sim - hardest_neg
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # PatchScoreModule interface
    # ------------------------------------------------------------------

    def compute_stop(self, parent_patch=None, **kwargs):
        """
        STOP score: contrastive margin at the current patch.
        Positive → patch is specific to pos_cancer_type.
        Negative → patch resembles a harder negative more than the target.
        """
        if parent_patch is None:
            return 0.0
        return self.weight * self._contrastive_score(parent_patch)

    def compute_zoom(self, parent_patch=None, child_patches=None, **kwargs):
        """
        ZOOM score: aggregated contrastive margin over child patches.
        """
        if not child_patches:
            return 0.0

        scores = []
        for p in child_patches:
            try:
                scores.append(self._contrastive_score(p))
            except Exception:
                continue

        if not scores:
            return 0.0

        agg_score = float(np.mean(scores)) if self.agg == "mean" else float(np.max(scores))
        return self.weight * agg_score

    def compute_diff(self, parent_patch=None, child_patches=None, **kwargs):
        """Gain in contrastive discriminability from zooming."""
        try:
            s_stop = self.compute_stop(parent_patch=parent_patch)
            s_zoom = self.compute_zoom(child_patches=child_patches)
            return s_zoom - s_stop
        except Exception:
            return 0.0

    def infer(self, s_stop, s_zoom):
        """Zoom if discriminability improves in children."""
        return 1 if s_zoom > s_stop else 0


# ==========================================================
# Registry
# ==========================================================
PATCH_SCORE_MODULES = {
    "img_sim_score": ImgSimScore,
    "text_align_score": TextAlignScore,
    "tissue_presence_score": TissuePresenceScore,
    "tissue_presence_penalty": TissuePresencePenalty,
    "entropy_score": EntropyScore,
    "cancer_centroid_score": CancerTypeCentroidScore,
    "contrastive_text_score": ContrastiveTextScore,
}

