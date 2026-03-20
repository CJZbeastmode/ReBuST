Read [](file:///Users/jay/Desktop/MA/timeline.md)

Read [](file:///Users/jay/Desktop/MA/data/pathology_keywords)

Read [](file:///Users/jay/Desktop/MA/data/pathology_keywords/default.json)

Read [](file:///Users/jay/Desktop/MA/src/downstream_tasks/wsi_classification/wsi_classification_plip.py#1-1), lines 1 to 100

Good context. Here are the technically elaborated answers to each of your points, grounded in what's in patch_scores.py, default.json, and your TCGA label structure.

---

## 1. Report-grounded reward → Structured cancer-type concept centroids

The interesting NLP extension beyond just using the slide's own report is building a **structured, label-conditional concept vocabulary**. You already have `labels_main.json` with per-slide TCGA type labels (LUAD, LUSC, COAD, READ, LIHC, etc.) and default.json as a generic flat keyword set. The direction:

For each cancer type in your dataset, curate a small (~10–15 phrase) set of *morphologically discriminative* text concepts — not generic pathology terms, but ones that a pathologist would specifically use for that diagnosis:

```
LUAD: ["acinar growth pattern", "lepidic spread", "mucin-producing adenocarcinoma", "glandular structures with Clara cells"]
LUSC: ["keratinization", "intercellular bridges", "squamous pearl formation", "nested squamous epithelium"]
COAD: ["tubular gland formation", "dirty necrosis", "goblet cells"]
LIHC: ["trabecular growth", "hepatocyte-like cells", "bile canaliculi", "sinusoidal vascular pattern"]
```

Pre-embed these with PLIP's text encoder and compute per-cancer-type centroids:
$$c_{\text{LUAD}} = \frac{1}{|K_{\text{LUAD}}|} \sum_{k \in K_{\text{LUAD}}} e_{\text{text}}(k)$$

Load the centroid for a given slide at construction time based on its label. The reward becomes:
$$r_{\text{stop}} = \cos(e_{\text{patch}},\ c_{\text{cancer\_type}})$$

This is cheap at runtime — centroid lookup is O(1). The interesting research question: **do cancer-type-specific centroids produce different patch selection behavior than the generic keyword average?** Concretely, does a LUAD slide have its glandular regions selected more reliably with the LUAD centroid versus default.json? You can measure this by checking how often the selected patches, when labeled by nearest-neighbor keyword, fall into the expected morphological categories.

For text-free inference: all of this is purely about constructing the right reward signal during training. At inference time, the A2C policy is already baked into the network weights — no text is ever needed when running on a new slide.

---

## 2. Contrastive prompt reward — full elaboration

The mathematical setup uses your known label structure to make reward discriminative between confusable cancer types. For each slide of type $y$, define:

$$r_{\text{contrastive}} = \underbrace{\cos(e_{\text{patch}},\ c_y)}_{\text{class-specific alignment}} - \underbrace{\max_{j \neq y}\ \cos(e_{\text{patch}},\ c_j)}_{\text{hardest negative alignment}}$$

The hardest negative can be pre-computed: LUAD vs LUSC are the hardest pair (both lung, different morphology). COAD vs READ is another. This mirrors **hard negative mining** from dense retrieval NLP literature (DPR, BEIR).

**Why this is stronger than `TextAlignScore`:** `TextAlignScore` is an absolute similarity — a patch showing necrotic tissue might score high against generic pathology terms even though necrosis is not diagnostically specific. The contrastive reward is zero-sum: a patch scores well only if it encodes features that are specific to the correct cancer, not just "looks pathological."

**Implementation as a `PatchScoreModule` subclass:**

```python
class ContrastiveTextScore(PatchScoreModule):
    def __init__(self, pos_cancer_type: str, neg_cancer_types: list[str],
                 concept_vocab: dict, embedder=None):
        # Pre-embed centroids once at init — no runtime text cost
        self.pos_centroid = self._embed_centroid(concept_vocab[pos_cancer_type], embedder)
        self.neg_centroids = [
            self._embed_centroid(concept_vocab[c], embedder)
            for c in neg_cancer_types
        ]

    def compute_stop(self, parent_patch=None, **kwargs):
        e = self.embedder.img_emb(parent_patch)
        pos_sim = cosine_similarity(e, self.pos_centroid)
        neg_sim = max(cosine_similarity(e, nc) for nc in self.neg_centroids)
        return float(pos_sim - neg_sim)     # margin

    def compute_zoom(self, parent_patch=None, child_patches=None, **kwargs):
        return float(np.mean([
            self._contrastive_score(self.embedder.img_emb(p))
            for p in child_patches
        ]))
```

This plugs directly into your `PATCH_SCORE_MODULES` registry and works unchanged with both HUMBE and the A2C reward pipeline.

**Research-worthy angle:** You can now ask: *does the hardness of the negative matter?* Run ablations:
- Easy negative: LUAD vs LIHC (different organ, trivially distinguishable visually)
- Hard negative: LUAD vs LUSC (same organ, different cell type)
- Hardest negative: LUAD vs LUSC + COAD (multi-class margin)

If hard negatives produce better downstream classification, that directly connects your patch selection method to the dense retrieval literature — a strong NLP framing.

---

## 3. Downstream task directions

**3a. Zero-shot slide retrieval as a second evaluation axis**

After A2C selects patches, mean-pool their embeddings: $\bar{e}_{\text{slide}} = \frac{1}{|S|} \sum_{p \in S} e_p$. Then rank all slides by cosine similarity to each cancer-type centroid and measure Recall@K. No fine-tuning — pure zero-shot retrieval. This gives you a second column in your results table that is a pure NLP metric (retrieval), decoupled from the classification head. The claim becomes: *if patch selection is semantically better, zero-shot retrieval improves, independently of whether TransformerMIL training benefits.*

**3b. Concept distribution shift analysis**

For every selected patch from every method, find its nearest keyword from default.json using cosine similarity in PLIP embedding space. Compute the empirical distribution over concept categories (tumor_morphology, necrosis, immune_infiltration, blank, etc.) per method. Then measure:
- **KL divergence** between methods' concept distributions
- **Blank/artifact fraction** as a proxy for selection quality
- Whether tumor_morphology fraction correlates with downstream AUROC

This is quantitative NLP analysis of what the visual policies "attend to", requires no additional annotation, and produces a novel table that no existing WSI paper has.

**3c. Concept-conditional classification (no new training)**

Compute separate slide vectors by averaging only patches whose nearest concept falls in a specific category. E.g., a slide vector from only tumor morphology patches vs. one from all patches. Test which subset-vector gives better TransformerMIL classification. This measures whether the NLP-grounded reward is steering selection toward the *right* semantic regions.

---

## On research worthiness

The thread connecting all of this is a single clean claim you can defend: **patch selection policy can be derived entirely from pre-trained visual-language supervision, with no patch-level annotation, and the choice of text signal (generic vs. cancer-specific vs. contrastive) measurably affects both selection behavior and downstream diagnostic accuracy.** That claim is falsifiable, has clear metrics (AUROC, Recall@K, concept distribution KL), and connects to VLP (vision-language pretraining), dense retrieval, and computational pathology literature simultaneously — which is exactly what puts it in NLP-for-medical territory rather than pure CV.