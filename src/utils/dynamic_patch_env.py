"""Module for dynamic patch env."""

import os
import numpy as np
from src.utils.patch_scores import *


class DynamicPatchEnv:
    """
    DynamicPatchEnv

    Reinforcement learning environment for hierarchical patch exploration
    on Whole Slide Images (WSIs).

    The environment is designed to be:
    - Fault-tolerant with respect to WSI I/O and embedding failures
    - Stable for long-running RL training
    - Semantically aligned with score-based STOP vs ZOOM decisions

    Actions:
        0 = STOP  → terminate episode at current location
        1 = ZOOM  → descend one pyramid level

    State representation:
        - Normalized spatial coordinates: (level, x, y)
        - PLIP image embedding of the current patch (512-D)

    Total state dimensionality: 515
    """

    def __init__(
        self,
        wsi,
        patch_score="text_align_score",
        patch_score_aggregation=None,
        image_class=None,
        patch_size=256,
        max_steps=8,
        backup_dir="data/supervised_dataset/env_backups",
        backup_interval=10,
    ):
        """
        Initialize the environment.

        Parameters
        ----------
        wsi : WSI
            Wrapper around an OpenSlide-compatible whole slide image
        patch_score : str
            Identifier of the patch scoring module used for reward computation
        patch_size : int
            Spatial size (in pixels) of extracted patches
        max_steps : int
            Maximum number of ZOOM actions per episode
        """

        self.wsi = wsi
        self.patch_size = patch_size
        self.max_steps = max_steps
        self.embedder = wsi.embedder
        self.image_class = image_class or self._resolve_image_class_from_wsi()

        # Pyramid bounds (coarse → fine)
        self.min_level = wsi.min_level
        self.max_level = wsi.max_level

        # Current environment state (trajectory-local)
        self.curr_level = None
        self.curr_x = None
        self.curr_y = None
        self.curr_patch = None  # Cached patch for embedding reuse
        self.steps = 0
        self.zoom_count = 0

        # Counts non-fatal environment failures for diagnostics
        self.env_error_count = 0

        # Optional backup directory (best-effort, non-critical)
        self.backup_dir = backup_dir
        self.backup_interval = backup_interval
        try:
            os.makedirs(self.backup_dir, exist_ok=True)
        except Exception:
            self.backup_dir = None

        # Patch scoring module (e.g. text-image alignment, entropy, etc.)
        score_cls = PATCH_SCORE_MODULES.get(patch_score)
        if score_cls is None:
            raise ValueError(f"Unknown patch_score module: {patch_score}")

        score_kwargs = {"embedder": self.embedder}
        if patch_score_aggregation is not None:
            score_kwargs["agg"] = patch_score_aggregation

        if patch_score == "cancer_centroid_score":
            if self.image_class is None:
                raise ValueError(
                    "cancer_centroid_score requires image class; could not resolve class from filename."
                )
            score_kwargs["cancer_type"] = self.image_class
        elif patch_score == "contrastive_text_score":
            if self.image_class is None:
                raise ValueError(
                    "contrastive_text_score requires image class; could not resolve class from filename."
                )
            score_kwargs["pos_cancer_type"] = self.image_class

        self.patch_score_module = score_cls(**score_kwargs)

        # print("Sampling root patches by density...")
        # self.root_patch_cache = self.sample_root_by_density(n_dense=4, n_mid=3, n_sparse=3)

        # Uncomment for debugging:
        # print(f"Initialized DynamicPatchEnv with patch score module: {patch_score}")

    # ---------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------
    def _resolve_image_class_from_wsi(self):
        """
        Resolve cancer class from slide filename.

        Expected naming example:
            TCGA-05-4390-LUAD.svs -> LUAD
        """
        image_path = getattr(self.wsi, "image_path", None)
        if not image_path:
            return None

        stem = os.path.splitext(os.path.basename(image_path))[0]
        normalized = stem.replace("_", "-")
        tokens = [tok.upper() for tok in normalized.split("-") if tok]

        for token in reversed(tokens):
            if token.isalpha() and 3 <= len(token) <= 6 and token != "TCGA":
                return token
        return None

    def _blank_state(self):
        """
        Return a valid zero-valued state vector.

        This serves as a last-resort fallback in case of unrecoverable
        errors (e.g. corrupt WSI tiles). The invariant that the state
        dimensionality remains constant is strictly preserved.
        """
        return np.zeros(3 + 512, dtype=np.float32)

    def _sample_root(self):
        """
        Sample a random starting location at the coarsest pyramid level.

        This encourages spatial diversity at episode initialization
        while keeping the initial observation computationally cheap.
        """
        # if len(self.root_patch_cache) > 0:
        #    ret_patch = self.root_patch_cache[0]
        #    self.root_patch_cache = self.root_patch_cache[1:]
        #    lvl, x, y, _ = ret_patch
        #    return lvl, x, y

        lvl = self.max_level
        W, H = self.wsi.levels_info[lvl]["size"]

        x = np.random.randint(0, max(1, W - self.patch_size))
        y = np.random.randint(0, max(1, H - self.patch_size))
        return lvl, x, y

    def sample_root_by_density(self, n_dense=4, n_mid=3, n_sparse=3):
        """
        Sample n_dense, n_mid, and n_sparse tissue patches from the root (coarsest) pyramid level based on tissue density.

        Dense: highest tissue density (most tissue)
        Sparse: lowest tissue density (least tissue)
        Mid: middle tissue density

        Returns:
            List of tuples: [(level, x, y, density_type)]
        """
        lvl = self.max_level
        W, H = self.wsi.levels_info[lvl]["size"]
        stride = self.patch_size
        coords = []
        # Grid sampling over the root level
        for x in range(0, W - stride + 1, stride):
            for y in range(0, H - stride + 1, stride):
                coords.append((x, y))

        density_patches = []
        for x, y in coords:
            try:
                patch = self.wsi.get_patch(lvl, x, y)
                # Calculate tissue density: simply count non-background pixels
                tissue_density = self.calculate_tissue_density(patch)
                density_patches.append((lvl, x, y, tissue_density))
            except Exception:
                continue

        # Sort patches by tissue density
        density_patches.sort(key=lambda tup: tup[3], reverse=True)

        # Select dense, sparse, and mid patches
        dense = density_patches[:n_dense]
        sparse = density_patches[-n_sparse:] if n_sparse > 0 else []
        mid_start = max(n_dense, (len(density_patches) - n_mid) // 2)
        mid = density_patches[mid_start : mid_start + n_mid] if n_mid > 0 else []

        # Annotate density type
        dense = [(lvl, x, y, "dense") for (lvl, x, y, _) in dense]
        mid = [(lvl, x, y, "mid") for (lvl, x, y, _) in mid]
        sparse = [(lvl, x, y, "sparse") for (lvl, x, y, _) in sparse]

        return dense + mid + sparse

    def calculate_tissue_density(self, patch):
        """
        Calculate the tissue density of a given patch. This function counts the number of non-background pixels
        in the patch to estimate tissue density.
        Returns a value between 0 (no tissue) and 1 (completely full of tissue).
        """
        # Convert patch to a binary mask (tissue vs. background)
        # Assuming a method `is_tissue_pixel` to check if a pixel is part of tissue
        tissue_pixels = sum(1 for pixel in patch if self.is_tissue_pixel(pixel))
        total_pixels = len(patch)

        # Return density as the proportion of tissue pixels
        return tissue_pixels / total_pixels if total_pixels > 0 else 0

    def is_tissue_pixel(self, pixel):
        """
        Check if a pixel is considered as tissue (non-background).
        This is just a placeholder, you might use a thresholding or color-based method.
        """
        # For now, assuming a simplistic threshold on the pixel value
        # Adjust this depending on how you define tissue in your images
        return pixel > 0  # Example: non-zero pixel is considered tissue

    def _safe_embed(self, patch):
        """
        Compute a PLIP embedding in a numerically safe manner.

        Any embedding failure (runtime error, NaNs, unexpected shapes)
        is handled locally and results in a zero vector. This prevents
        silent corruption of the replay buffer and downstream gradients.
        """
        try:
            emb = self.embedder.img_emb(patch).numpy()
        except Exception:
            return np.zeros(512, dtype=np.float32)

        e = np.asarray(emb)

        # Normalize common output shapes
        if e.ndim == 1:
            emb1 = e
        elif e.ndim == 2:
            emb1 = e[0] if e.shape[0] == 1 else e.mean(axis=0)
        else:
            emb1 = e.flatten()

        emb1 = emb1.astype(np.float32)

        # Enforce fixed embedding dimensionality
        if emb1.size < 512:
            emb1 = np.pad(emb1, (0, 512 - emb1.size))
        elif emb1.size > 512:
            emb1 = emb1[:512]

        # Explicit numerical stability check
        if not np.isfinite(emb1).all():
            emb1 = np.zeros(512, dtype=np.float32)

        return emb1

    def _get_state(self, patch=None):
        """
        Construct the RL state vector.

        The method is exception-safe by design and guarantees that a
        valid state vector is always returned, even if patch access
        or embedding computation fails.
        """
        try:
            W, H = self.wsi.levels_info[self.curr_level]["size"]

            coords = np.array(
                [
                    self.curr_level / max(1, self.max_level),
                    self.curr_x / max(1, W),
                    self.curr_y / max(1, H),
                ],
                dtype=np.float32,
            )

            if self.embedder is not None and patch is not None:
                emb = self._safe_embed(patch)
            elif self.embedder is not None and self.curr_patch is not None:
                emb = self._safe_embed(self.curr_patch)
            else:
                emb = np.zeros(512, dtype=np.float32)

            return np.concatenate([coords, emb])

        except Exception:
            # Absolute fallback: environment must not crash
            return self._blank_state()

    def calculate_score(self, parent_level: int, parent_x: int, parent_y: int):
        """
        Compute STOP and ZOOM scores for a given parent patch location.

        This method is primarily used for analysis, ablations, and
        standalone inference outside the RL training loop.

        It evaluates:
        - STOP score at the parent location
        - ZOOM score based on all valid child patches

        Returns
        -------
        valid : bool
            Indicates whether score computation was successful
        scores : list[float]
            [s_stop, s_zoom]
        """
        if parent_level <= self.min_level:
            # terminal: cannot zoom further
            try:
                parent_patch = self.wsi.get_patch(parent_level, parent_x, parent_y)
                r_stop = self.patch_score_module.compute_stop(
                    parent_patch=parent_patch,
                    image_class=self.image_class,
                )
            except Exception:
                return False, 0.0

            return True, [r_stop, -np.inf]

        parent_x = int(parent_x)
        parent_y = int(parent_y)

        try:
            parent_patch = self.wsi.get_patch(parent_level, parent_x, parent_y)
        except Exception:
            return False, 0.0

        child_grids = self.wsi.get_child_grid(parent_level, parent_x, parent_y)

        child_patches = []
        child_coords = []

        for grid in child_grids:
            for child_x, child_y in grid:
                try:
                    patch = self.wsi.get_patch(parent_level - 1, child_x, child_y)
                except Exception:
                    continue
                child_patches.append(patch)
                child_coords.append((child_x, child_y))

        if len(child_patches) == 0:
            r_stop = self.patch_score_module.compute_stop(
                parent_patch=parent_patch,
                image_class=self.image_class,
            )
            r_zoom = self.patch_score_module.compute_zoom(
                parent_patch=parent_patch,
                child_patches=[],
                image_class=self.image_class,
            )
            return True, [r_stop, r_zoom]

        r_stop = self.patch_score_module.compute_stop(
            parent_patch=parent_patch,
            image_class=self.image_class,
        )
        r_zoom = self.patch_score_module.compute_zoom(
            parent_patch=parent_patch,
            child_patches=child_patches,
            image_class=self.image_class,
        )

        return True, [r_stop, r_zoom]

    def infer_zoom_decision(self, s_stop, s_zoom):
        """
        Infer a STOP vs ZOOM decision from precomputed scores.

        This delegates the decision logic to the configured patch
        scoring module and is typically used for greedy inference.
        """
        return self.patch_score_module.infer(s_stop, s_zoom)

    # ---------------------------------------------------------
    # Encoding for standalone inference
    # ---------------------------------------------------------
    def encode_state(self, patch, lvl=None, x=None, y=None):
        """
        Encode a patch into a state vector without stepping the environment.

        This method is intended for inference and visualization pipelines
        where state construction is required independently of RL dynamics.
        """
        if lvl is not None:
            self.curr_level = lvl
        if x is not None:
            self.curr_x = x
        if y is not None:
            self.curr_y = y

        self.curr_patch = patch

        W, H = self.wsi.levels_info[self.curr_level]["size"]
        level_norm = self.curr_level / self.max_level
        x_norm = self.curr_x / W
        y_norm = self.curr_y / H

        coords = np.array([level_norm, x_norm, y_norm], dtype=np.float32)

        if self.embedder is not None:
            emb1 = self._safe_embed(patch)
            return np.concatenate([coords, emb1])
        else:
            return np.concatenate([coords, np.zeros(512, dtype=np.float32)])

    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------
    def reset(self):
        """
        Reset the environment at the beginning of a new episode.

        This method is exception-safe and guarantees a valid initial state.
        """
        self.steps = 0
        self.zoom_count = 0

        try:
            self.curr_level, self.curr_x, self.curr_y = self._sample_root()
            self.curr_patch = self.wsi.get_patch(
                self.curr_level, self.curr_x, self.curr_y
            )
            return self._get_state(self.curr_patch)
        except Exception as e:
            self.env_error_count += 1
            print(f"[ENV RESET WARNING] {e}")
            return self._blank_state()

    # ---------------------------------------------------------
    # Step
    # ---------------------------------------------------------
    def step(self, action):
        """
        Execute a single environment step.

        Any failure during patch access or scoring results in a soft
        episode termination with a negative reward.
        """
        try:
            return self._step_impl(action)
        except Exception as e:
            self.env_error_count += 1
            print(f"[ENV STEP WARNING #{self.env_error_count}] {e}")
            return self._blank_state(), -1.0, True, {"type": "env_failure"}

    def _step_impl(self, action):
        """
        Internal step logic implementing STOP vs ZOOM transitions.

        Core principles of this implementation:
        ---------------------------------------
        1. Reward depends ONLY on the chosen action (MDP requirement)
        2. Relative preference (zoom vs stop) is delegated to PatchScoreModule
        3. ZOOM cost is NOT baked into reward (handled externally by RL)
        4. Transition stochasticity is minimized
        """

        done = False
        info = {}

        # ---------------------------------------------------------
        # 1. Get parent patch (current state)
        # ---------------------------------------------------------
        try:
            parent_patch = self.wsi.get_patch(self.curr_level, self.curr_x, self.curr_y)
        except Exception:
            # unrecoverable → terminate episode
            return self._blank_state(), -1.0, True, {"invalid": True}

        # ---------------------------------------------------------
        # 2. Compute STOP score (always available)
        # ---------------------------------------------------------
        try:
            s_stop = float(
                self.patch_score_module.compute_stop(
                    parent_patch=parent_patch,
                    image_class=self.image_class,
                )
            )
        except Exception:
            s_stop = 0.0

        # ---------------------------------------------------------
        # 3. Compute ZOOM score (only if zoom is possible)
        # ---------------------------------------------------------
        s_zoom = None
        child_patches = []
        child_coords = []

        if self.curr_level > self.min_level:
            child_level = self.curr_level - 1

            try:
                grids = self.wsi.get_child_grid(
                    self.curr_level, self.curr_x, self.curr_y
                )
            except Exception:
                grids = []

            for grid in grids:
                for cx, cy in grid:
                    try:
                        patch = self.wsi.get_patch(child_level, cx, cy)
                        child_patches.append(patch)
                        child_coords.append((cx, cy))
                    except Exception:
                        continue

            if child_patches:
                try:
                    s_zoom = float(
                        self.patch_score_module.compute_zoom(
                            parent_patch=parent_patch,
                            child_patches=child_patches,
                            image_class=self.image_class,
                        )
                    )
                except Exception:
                    s_zoom = 0.0

        # ---------------------------------------------------------
        # 4. Execute action
        # ---------------------------------------------------------
        if action == 0:  # STOP
            # ---------------------------------------------
            # Reward is STOP score only
            # ---------------------------------------------
            reward = s_stop
            done = True
            next_state = None
            action_name = "STOP"

        else:  # ZOOM
            if s_zoom is None or not child_patches:
                # Cannot zoom → forced termination
                reward = s_stop
                done = True
                next_state = None
                action_name = "FORCED_STOP"

            else:
                # -----------------------------------------
                # Select next child deterministically
                # (minimizes transition noise)
                # -----------------------------------------
                idx = np.argmax(
                    [
                        self.patch_score_module.compute_stop(
                            parent_patch=p,
                            image_class=self.image_class,
                        )
                        for p in child_patches
                    ]
                )

                self.curr_level -= 1
                self.curr_x, self.curr_y = child_coords[idx]
                self.curr_patch = child_patches[idx]

                self.steps += 1
                self.zoom_count += 1

                reward = s_zoom
                action_name = "ZOOM"

                if self.steps >= self.max_steps:
                    done = True
                    next_state = None
                else:
                    next_state = self._get_state(self.curr_patch)

        # ---------------------------------------------------------
        # 5. Optional: score diagnostics (NOT used for reward)
        # ---------------------------------------------------------
        try:
            s_diff = float(
                self.patch_score_module.compute_diff(
                    s_stop=s_stop,
                    s_zoom=s_zoom,
                    image_class=self.image_class,
                )
            )
        except Exception:
            s_diff = None

        info.update(
            {
                "action": action_name,
                "s_stop": s_stop,
                "s_zoom": s_zoom,
                "s_diff": s_diff,
                "level": self.curr_level,
                "zoom_count": self.zoom_count,
            }
        )

        # ---------------------------------------------------------
        # 6. Final reward normalization (optional but consistent)
        # ---------------------------------------------------------
        reward = float(np.clip(reward, -1.0, 1.0))

        return next_state, reward, done, info
