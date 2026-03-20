import os
import json
import base64
from io import BytesIO
import openslide
from PIL import Image
import torch
import math
from typing import Iterator

from .embedder import Embedder


class WSI:
    def __init__(
        self,
        image_path,
        patch_size=256,
        target_min_side=512,
        synthetic_scale=0.5,
        max_level_side=40000,
        embedder=None,
        img_embedding_backend="plip",
        min_level: int | None = None,
        multistage: bool = False,
    ):
        """
        Load the WSI, initialize PLIP, generate synthetic pyramid levels.

        At init, all patches start at ``max_level`` (coarsest), identical to
        the original WSI behavior.  Per-patch levels are tracked via the
        ``active_patches`` hashmap; individual patches can be zoomed to finer
        levels independently after construction.

        Parameters
        ----------
        min_level : int | None
            Floor on how far any patch may be zoomed.  ``None`` (default) uses
            the finest non-frozen native level (typically level 0 / 1).  Pass
            a positive integer (e.g. ``2``) to prevent A2C / HUMBE from ever
            descending past that level.
        multistage : bool
            When ``True`` (multistage / HUMBE-B mode), newly initialised patches
            have ``zoomable=False`` by default — the budget enforcer decides which
            patches are eligible for refinement.
            When ``False`` (default, single-stage A2C / greedy mode), patches are
            initialised with ``zoomable=True`` so any patch may be zoomed freely.
        """

        # OpenSlide — fail fast on unsupported / corrupt files.
        self.slide = openslide.OpenSlide(image_path)

        self.patch_size = patch_size
        self.target_min_side = target_min_side
        self.synthetic_scale = synthetic_scale
        self.max_level_side = max_level_side
        self.multistage = multistage

        # Track level info
        self.levels_info = {}

        # Store synthetic images
        self.synthetic_images = {}

        # Load PLIP / CONCH embedder
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        if embedder is None:
            if img_embedding_backend == "plip":
                embedder = Embedder(img_backend="plip", device=device)
            elif img_embedding_backend == "conch":
                embedder = Embedder(img_backend="conch", device=device)
            else:
                embedder = Embedder()
        # Keep the Embedder instance for encoding convenience
        self.embedder = embedder
        self.model = embedder.model
        self.processor = embedder.processor

        self.device = device

        # Build native levels
        self._build_native_levels()

        # Set min_level to the finest non-frozen native level.
        # Level 0 (native full-res) is typically frozen (too large to tile), so
        # min_level ends up as 1 (or wherever the first accessible level is).
        # This prevents A2C from zooming into frozen levels that can't be read.
        self.min_level = min(
            lvl
            for lvl, info in self.levels_info.items()
            if not info.get("frozen", False)
        )

        # Generate synthetic levels at the coarse end
        self._generate_synthetic_levels()

        # Apply optional user-specified floor (must be >= computed min_level)
        if min_level is not None:
            clamped = max(self.min_level, min(int(min_level), self.max_level - 1))
            if clamped != self.min_level:
                print(f"[WSI] min_level set to {clamped} (user override)")
            self.min_level = clamped

        # ------------------------------------------------------------------
        # Per-patch level tracking
        # ------------------------------------------------------------------
        # Key  : (level, x, y) — absolute patch coordinates at its current level.
        # Value: free-form metadata dict (score, provenance, …) — empty by default.
        self.active_patches: dict[tuple[int, int, int], dict] = {}

        # Interior nodes that were expanded (replaced by children).
        # Stored as a dict so we can attach metadata (e.g. zoomable=True).
        # Kept separately so the visualiser can render zoomed-past regions.
        self.zoomed_patches: dict[tuple[int, int, int], dict] = {}

        # Default: flat grid of all patches at max_level — same as original WSI.
        self._init_root_patches()

    # =====================================================================
    # Build native levels (unchanged from original)
    # =====================================================================
    def _build_native_levels(self):
        """Add all native OpenSlide levels to levels_info."""
        self.min_level = 0  # will be updated

        for lvl in range(self.slide.level_count):
            w, h = self.slide.level_dimensions[lvl]

            frozen = max(w, h) > self.max_level_side

            if not frozen and lvl < getattr(self, "min_level", lvl):
                self.min_level = lvl

            pw = int(math.ceil(w / float(self.patch_size)) * self.patch_size)
            ph = int(math.ceil(h / float(self.patch_size)) * self.patch_size)

            self.levels_info[lvl] = {
                "size": (w, h),
                "padded_size": (pw, ph),
                "type": "native",
                "native_idx": lvl,
                "downsample": self.slide.level_downsamples[lvl],
                "frozen": frozen,
            }

    # =====================================================================
    # Generate synthetic coarse levels
    # =====================================================================
    def _generate_synthetic_levels(self):
        """
        Create downsampled synthetic levels beyond the coarsest native level.
        Uses consistent 0.5 scale factor for synthetic levels.
        """

        # Find the coarsest native level
        max_native_idx = self.slide.level_count - 1

        # If the coarsest native level is frozen, do not generate synthetic levels
        if self.levels_info[max_native_idx].get("frozen", False):
            self.max_level = max_native_idx
            return

        base_img = self.slide.read_region(
            (0, 0), max_native_idx, self.slide.level_dimensions[max_native_idx]
        ).convert("RGB")

        w, h = base_img.size
        scale = self.synthetic_scale
        current_level = max_native_idx + 1

        while True:
            new_w, new_h = int(w * scale), int(h * scale)

            # Stop generating synthetic levels once resolution becomes
            # too small to yield meaningful patch-level features.
            if min(new_w, new_h) < self.target_min_side:
                break

            # Resize to create synthetic level
            synth = base_img.resize((new_w, new_h), Image.BILINEAR)

            # Pad synthetic image so its dimensions are multiples of patch_size
            pw = int(math.ceil(new_w / float(self.patch_size)) * self.patch_size)
            ph = int(math.ceil(new_h / float(self.patch_size)) * self.patch_size)
            if (pw, ph) != (new_w, new_h):
                canvas = Image.new("RGB", (pw, ph), (255, 255, 255))
                canvas.paste(synth, (0, 0))
                synth = canvas

            self.synthetic_images[current_level] = synth

            self.levels_info[current_level] = {
                "size": (new_w, new_h),
                "padded_size": (pw, ph),
                "type": "synthetic",
                "frozen": False,
            }

            base_img = synth
            w, h = new_w, new_h
            current_level += 1

        self.max_level = max(self.levels_info.keys())

    # =====================================================================
    # Patch extraction
    # =====================================================================
    def get_patch(self, lvl_id, x, y):
        """
        Extract patch from either real level or synthetic level.
        Raises exception if patch cannot be read (corrupted file).
        Output: img (PIL Image)
        """
        x = int(x)
        y = int(y)

        entry = self.levels_info[lvl_id]

        if entry.get("frozen", False):
            raise RuntimeError(
                f"Level {lvl_id} is frozen (size {entry['size']} exceeds limit)"
            )

        # use padded size for tiling / bounds when available
        if "padded_size" in entry:
            w, h = entry["padded_size"]
        else:
            w, h = entry["size"]

        # native WSI level
        if entry["type"] == "native":
            native_idx = entry["native_idx"]
            try:
                ds = self.slide.level_downsamples[native_idx]
                lx = int(x * ds)
                ly = int(y * ds)

                img = self.slide.read_region(
                    (lx, ly), native_idx, (self.patch_size, self.patch_size)
                ).convert("RGB")
                # Ensure patch is exactly patch_size x patch_size; pad with white if smaller
                if img.size != (self.patch_size, self.patch_size):
                    canvas = Image.new(
                        "RGB", (self.patch_size, self.patch_size), (255, 255, 255)
                    )
                    canvas.paste(img, (0, 0))
                    img = canvas
                return img
            except Exception as e:
                # ADDED COMMENT:
                # OpenSlide/OpenJPEG failures are surfaced explicitly
                # so callers (e.g. RL env) can catch and skip safely.
                raise RuntimeError(
                    f"Failed to read patch at level {lvl_id}, pos ({x},{y}): {e}"
                )

        # synthetic precomputed level
        elif entry["type"].lower() == "synthetic":
            img = self.synthetic_images[lvl_id]

            # Clamp start coordinates to padded image so cropping is safe.
            # This ensures patches that lie beyond original content are pure white.
            padded_w, padded_h = entry.get("padded_size") or entry["size"]
            max_x = max(0, padded_w - self.patch_size)
            max_y = max(0, padded_h - self.patch_size)
            x = max(0, min(x, max_x))
            y = max(0, min(y, max_y))

            x2 = x + self.patch_size
            y2 = y + self.patch_size

            patch = img.crop((x, y, x2, y2)).convert("RGB")
            # crop should be exactly patch_size due to padding, but keep safety pad
            if patch.size != (self.patch_size, self.patch_size):
                canvas = Image.new(
                    "RGB", (self.patch_size, self.patch_size), (255, 255, 255)
                )
                canvas.paste(patch, (0, 0))
                patch = canvas
            return patch

        else:
            raise ValueError(f"Unknown level type: {entry['type']}")

    # =====================================================================
    # Dynamic scale between levels
    # =====================================================================
    def get_scale(self, parent_level):
        """
        Compute scale factor from parent level to child (next finer) level.
        Scale = child_width / parent_width

        For native levels, this can be 2, 4, or other values.
        For synthetic levels, this is typically 2.
        """
        child_level = parent_level - 1
        if child_level < 0:
            return None

        pw, _ = self.levels_info[parent_level]["size"]
        cw, _ = self.levels_info[child_level]["size"]

        return cw / pw

    def get_num_children(self, parent_level):
        """
        Compute the number of child patches that fit in one parent patch.

        If scale = 2, we get 2x2 = 4 children.
        If scale = 4, we get 4x4 = 16 children.

        Returns (num_children_per_side, total_children)
        """
        scale = self.get_scale(parent_level)
        if scale is None:
            return None, None

        # Round to nearest integer for grid calculation
        num_per_side = int(round(scale))
        total = num_per_side * num_per_side

        return num_per_side, total

    def get_child_grid(self, parent_level, parent_x=None, parent_y=None):
        """
        Return child-patch coordinates relative to a parent patch.

        Calling modes
        -------------
        ``get_child_grid(parent_level)``
            Legacy / offset mode — returns ``List[(dx, dy)]``, the relative
            offsets (in child-level pixels) for each child of any parent patch
            at ``parent_level``.  Used by single-stage inference helpers that
            only need the structural grid without a specific parent location.

        ``get_child_grid(parent_level, parent_x, parent_y)``
            Absolute mode — returns ``List[List[(nx, ny)]]``, one inner list
            per child, each containing the absolute child-level coordinate of
            that child.  Used by multistage / WSI_B-style callers.

        Returns
        -------
        List[(dx, dy)]  when parent_x/parent_y are None
        List[List[(nx, ny)]]  otherwise
        """
        scale = self.get_scale(parent_level)
        if scale is None:
            return []

        num_per_side = int(round(scale))
        child_level = parent_level - 1
        if child_level < 0:
            return []

        # --- offset-only mode (legacy) ---
        if parent_x is None or parent_y is None:
            offsets = []
            for row in range(num_per_side):
                for col in range(num_per_side):
                    dx = col * self.patch_size
                    dy = row * self.patch_size
                    offsets.append((dx, dy))
            return offsets

        # --- absolute coordinate mode ---
        cx = int(parent_x * scale)
        cy = int(parent_y * scale)

        child_grids = []
        for row in range(num_per_side):
            for col in range(num_per_side):
                nx = cx + col * self.patch_size
                ny = cy + row * self.patch_size
                child_grids.append([(nx, ny)])

        return child_grids

    # =====================================================================
    # Get embedding
    # =====================================================================
    def get_emb(self, img):
        """Get PLIP image embedding."""
        # Delegate to the central Embedder implementation which handles
        # PIL/numpy/torch inputs, blank-patch checks, normalization and caching.
        return self.embedder.img_emb(img)

    # =========================================================================
    # Patch iterator
    # =========================================================================

    def iterate_patches(self, lvl_id) -> Iterator[tuple[int, int]]:
        """Yield every (x, y) patch coordinate for the given pyramid level."""
        entry = self.levels_info[lvl_id]
        if entry.get("frozen", False):
            return
        w, h = entry.get("padded_size") or entry["size"]
        for y in range(0, h, self.patch_size):
            for x in range(0, w, self.patch_size):
                yield x, y

    # =========================================================================
    # Per-patch level tracking
    # =========================================================================

    def _init_root_patches(self):
        """
        Populate active_patches with all patches at max_level.

        Replicates the original WSI flat-grid behaviour: every patch starts at
        the coarsest level.

        * ``multistage=False`` — ``zoomable`` is
          never stored; every patch is unconditionally zoomable.  Use
          ``is_zoomable()`` to query.
        * ``multistage=True`` — ``zoomable=False`` stored
          so the budget enforcer is the sole authority on which patches may be
          zoomed.
        """
        self.active_patches.clear()
        self.zoomed_patches.clear()
        if self.multistage:
            for x, y in self.iterate_patches(self.max_level):
                self.active_patches[(self.max_level, x, y)] = {"zoomable": False}
        else:
            for x, y in self.iterate_patches(self.max_level):
                self.active_patches[(self.max_level, x, y)] = {}

    def zoom_patch(
        self, lvl: int, x: int, y: int
    ) -> list[tuple[int, int, int]]:
        """
        Replace the active patch at (lvl, x, y) with its children.

        The parent is removed from active_patches and recorded in
        zoomed_patches.  All children are inserted into active_patches
        with default metadata (inheriting the multistage zoomable default).

        Parameters
        ----------
        lvl, x, y : int
            Coordinates of the patch to zoom.

        Returns
        -------
        list of (child_level, cx, cy)
            Keys of all newly added child patches.

        Raises
        ------
        KeyError  : patch is not currently active.
        ValueError: already at min_level (cannot zoom further).
        """
        key = (lvl, x, y)
        if key not in self.active_patches:
            raise KeyError(f"Patch {key} is not in active_patches")
        if lvl <= self.min_level:
            raise ValueError(
                f"Cannot zoom patch at min_level {self.min_level}"
            )

        # Move parent: active → zoomed.
        # In multistage mode, zoomable=False marks it as an interior node.
        # In single-stage mode, zoomable is never stored.
        parent_meta = self.active_patches.pop(key)
        if self.multistage:
            self.zoomed_patches[key] = {**parent_meta, "zoomable": False}
        else:
            self.zoomed_patches[key] = {k: v for k, v in parent_meta.items() if k != "zoomable"}

        child_level = lvl - 1
        added: list[tuple[int, int, int]] = []
        for grid in self.get_child_grid(lvl, x, y):
            for cx, cy in grid:
                child_key = (child_level, cx, cy)
                if self.multistage:
                    self.active_patches[child_key] = {"parent": key, "zoomable": False}
                else:
                    self.active_patches[child_key] = {"parent": key}
                added.append(child_key)

        return added

    def load_from_humbe(
        self,
        selected_coords: list[tuple[int, int, int]],
        zoomed_coords: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """
        Replace the active set with the output of HUMBE (or any external
        hierarchical budget enforcer).

        Parameters
        ----------
        selected_coords : List[(lvl, x, y)]
            Final leaf patches that HUMBE decided to keep.
        zoomed_coords : List[(lvl, x, y)], optional
            Intermediate patches that HUMBE expanded.  Recorded in
            zoomed_patches so the visualiser can render them separately.
        """
        self.active_patches = {
            (lvl, x, y): {} for lvl, x, y in selected_coords
        }
        self.zoomed_patches = {
            (lvl, x, y): {} for lvl, x, y in (zoomed_coords or [])
        }

    def set_patch_metadata(self, lvl: int, x: int, y: int, metadata: dict) -> None:
        """
        Merge ``metadata`` into the stored dict for an active or zoomed patch.

        Uses dict.update() semantics: existing keys not present in ``metadata``
        are preserved.  Raises KeyError if the patch is in neither dict.

        When ``multistage=False`` the ``zoomable`` key is silently stripped
        from ``metadata`` before storing — zoom eligibility is unconditional
        in single-stage mode and must not be overridden via metadata.
        """
        if not self.multistage:
            metadata = {k: v for k, v in metadata.items() if k != "zoomable"}
        key = (lvl, x, y)
        if key in self.active_patches:
            self.active_patches[key].update(metadata)
        elif key in self.zoomed_patches:
            self.zoomed_patches[key].update(metadata)
        else:
            raise KeyError(f"Patch {key} is neither active nor zoomed")

    def get_patch_metadata(self, lvl: int, x: int, y: int) -> dict:
        """Return the metadata dict for an active or zoomed patch, or {} if not found."""
        key = (lvl, x, y)
        if key in self.active_patches:
            return self.active_patches[key]
        return self.zoomed_patches.get(key, {})

    def is_zoomable(self, lvl: int, x: int, y: int) -> bool:
        """
        Return whether the patch at (lvl, x, y) is eligible to be zoomed.

        * ``multistage=False``: always ``True`` — zoom eligibility is
          unconditional in single-stage mode; the ``zoomable`` metadata key is
          never consulted.
        * ``multistage=True``: reads the ``zoomable`` flag from the patch's
          metadata dict.  Defaults to ``True`` if the key is
          absent (safe for virtual children that carry no metadata).
        """
        if not self.multistage:
            return True
        return self.get_patch_metadata(lvl, x, y).get("zoomable", True)

    def is_active(self, lvl: int, x: int, y: int) -> bool:
        """Return True if (lvl, x, y) is a current leaf in active_patches."""
        return (lvl, x, y) in self.active_patches

    def get_active_patches(self) -> list[tuple[int, int, int]]:
        """Return a sorted list of all active (level, x, y) keys."""
        return sorted(self.active_patches.keys())

    def active_patch_count(self) -> int:
        """Return the number of currently active patches."""
        return len(self.active_patches)

    def dump_zoomable_grid(
        self,
        output_path: str = "./data/visualizations/property.html",
        title: str = "Zoomable grid",
    ) -> str:
        """
        Write an HTML file showing the ``zoomable`` flag for every patch
        position at every pyramid level.

        Every cell shows "T" (green, zoomable=True) or "F" (red, zoomable=False
        or info not present).

        Parameters
        ----------
        output_path : str
            Path to write the HTML file.
        title : str
            Heading text shown at the top of the page.

        Returns
        -------
        str  Path to the saved file.
        """
        all_levels: set[int] = set()
        for lvl, info in self.levels_info.items():
            if not info.get("frozen", False):
                all_levels.add(lvl)
        for lvl, _, _ in self.active_patches:
            all_levels.add(lvl)
        for lvl, _, _ in self.zoomed_patches:
            all_levels.add(lvl)

        ps = self.patch_size

        rows_html: list[str] = []
        for lvl in sorted(all_levels, reverse=True):
            info = self.levels_info.get(lvl, {})
            if info.get("frozen", False):
                continue

            w, h = info.get("padded_size") or info.get("size") or (0, 0)
            if w == 0 or h == 0:
                continue

            cols = max(1, (w + ps - 1) // ps)
            rows = max(1, (h + ps - 1) // ps)

            n_active = sum(1 for (l, _, _) in self.active_patches if l == lvl)
            n_zoomed = sum(1 for (l, _, _) in self.zoomed_patches if l == lvl)

            cell: dict[tuple[int, int], bool] = {}

            for (l, x, y), meta in self.active_patches.items():
                if l != lvl:
                    continue
                cell[(x // ps, y // ps)] = bool(meta.get("zoomable", False))

            for (l, x, y), meta in self.zoomed_patches.items():
                if l != lvl:
                    continue
                cell[(x // ps, y // ps)] = bool(meta.get("zoomable", False))

            table_rows: list[str] = []
            for r in range(rows):
                tds: list[str] = []
                for c in range(cols):
                    zoomable = cell.get((c, r), False)
                    if zoomable:
                        tds.append('<td class="t">T</td>')
                    else:
                        tds.append('<td class="f">F</td>')
                table_rows.append(f"<tr>{''.join(tds)}</tr>")

            level_type = info.get("type", "?")
            rows_html.append(f"""
<section>
  <h2>Level {lvl} &nbsp;<small>({level_type}, {cols}&thinsp;&times;&thinsp;{rows} patches &mdash; active: {n_active}, zoomed: {n_zoomed})</small></h2>
  <div class="scroll">
  <table>
    {''.join(table_rows)}
  </table>
  </div>
</section>""")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: monospace; background: #111; color: #eee; padding: 1em; }}
  h1   {{ color: #fff; }}
  h2   {{ color: #aaa; margin-top: 1.5em; border-bottom: 1px solid #333; padding-bottom: 0.2em; }}
  small {{ font-size: 0.7em; color: #888; }}
  .scroll {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; }}
  td {{
    width: 22px; height: 22px;
    text-align: center; vertical-align: middle;
    font-size: 11px; font-weight: bold;
    border: 1px solid #2a2a2a;
  }}
  td.t {{ background: #1a6e1a; color: #afffaf; }}
  td.f {{ background: #6e1a1a; color: #ffafaf; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p>
  <span style="background:#1a6e1a;color:#afffaf;padding:2px 6px">T</span> zoomable=True &nbsp;
  <span style="background:#6e1a1a;color:#ffafaf;padding:2px 6px">F</span> zoomable=False (or not present)
</p>
{''.join(rows_html)}
</body>
</html>"""

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        totals = {
            "active": len(self.active_patches),
            "zoomed": len(self.zoomed_patches),
        }
        print(f"[WSI zoomable-grid] Saved \u2192 {output_path}  "
              f"(active={totals['active']}, zoomed={totals['zoomed']})")
        return output_path

    def reset(self) -> None:
        """
        Reset to the initial flat-grid state: all patches at max_level.
        Equivalent to calling _init_root_patches().
        """
        self._init_root_patches()

    def visualize(
        self,
        output_html: str = "./data/visualizations/visualization.html",
        metadata: dict | None = None,
    ) -> str:
        """
        Render an HTML visualization of the current active / zoomed patch state.

        Parameters
        ----------
        output_html : str
            Path to write the HTML file.
        metadata : dict, optional
            Key-value pairs shown as a header in the sidebar.
            Example: {"Method": "HUMBE", "Image": "slide.svs", "Budget": "25%"}

        Returns
        -------
        str
            Path to the saved HTML file.
        """
        # -----------------------------------------------------------------
        # 1. WSI thumbnail (always at max_level for the overview)
        # -----------------------------------------------------------------
        thumb_level = self.max_level
        thumb_w, thumb_h = self.levels_info[thumb_level]["size"]

        level_entry = self.levels_info[thumb_level]
        if level_entry.get("type", "native").lower() == "synthetic":
            # Synthetic levels are stored as PIL images in self.synthetic_images
            thumbnail = self.synthetic_images[thumb_level].convert("RGB")
            thumb_w, thumb_h = thumbnail.size
        else:
            # Native level: use the OpenSlide native_idx, not the internal level id
            native_idx = level_entry["native_idx"]
            thumbnail = self.slide.read_region(
                (0, 0), native_idx, (thumb_w, thumb_h)
            ).convert("RGB")

        buf = BytesIO()
        thumbnail.save(buf, format="PNG")
        img_base64 = base64.b64encode(buf.getvalue()).decode("ascii")

        display_scale = 1.0
        display_width = int(thumb_w * display_scale)
        display_height = int(thumb_h * display_scale)

        # -----------------------------------------------------------------
        # 2. Coordinate mapping helpers
        # -----------------------------------------------------------------
        level0_w, _ = self.levels_info[0]["size"]

        ds: dict[int, float] = {}
        for lvl, info in self.levels_info.items():
            w_l, _ = info["size"]
            ds[lvl] = level0_w / float(w_l) if w_l > 0 else 1.0

        ds_thumb = ds[thumb_level]

        def to_thumb(lvl: int, x: int, y: int, patch_size: int):
            ratio = ds[lvl] / ds_thumb
            X = int(x * ratio * display_scale)
            Y = int(y * ratio * display_scale)
            S = max(1, int(patch_size * ratio * display_scale))
            return X, Y, S

        # -----------------------------------------------------------------
        # 3. Level depth → shade mapping for active patches
        # -----------------------------------------------------------------
        depth_range = max(1, self.max_level - self.min_level)

        def active_color(lvl: int) -> str:
            depth = self.max_level - lvl
            t = depth / depth_range
            r = int(144 * (1 - t))
            g = int(238 - (238 - 100) * t)
            b = int(144 * (1 - t))
            return f"#{r:02X}{g:02X}{b:02X}"

        def active_border(lvl: int) -> str:
            depth = self.max_level - lvl
            t = depth / depth_range
            r = int(80 * (1 - t))
            g = int(130 - 130 * t)
            b = int(80 * (1 - t))
            return f"#{r:02X}{g:02X}{b:02X}"

        zoomed_fill = "#FF4444"
        zoomed_border = "#880000"

        # -----------------------------------------------------------------
        # 4. Build overlay list
        # -----------------------------------------------------------------
        overlays = []
        active_counts: dict[int, int] = {}
        zoomed_counts: dict[int, int] = {}

        for lvl, x, y in self.active_patches:
            X, Y, S = to_thumb(lvl, x, y, self.patch_size)
            meta = self.active_patches[(lvl, x, y)]
            score_str = f"{meta.get('score', '')}" if meta else ""
            active_counts[lvl] = active_counts.get(lvl, 0) + 1
            overlays.append({
                "x": X, "y": Y, "size": S,
                "level": lvl,
                "type": "active",
                "orig_x": x, "orig_y": y,
                "score": score_str,
                "fill": active_color(lvl),
                "border": active_border(lvl),
            })

        for lvl, x, y in self.zoomed_patches:
            X, Y, S = to_thumb(lvl, x, y, self.patch_size)
            zoomed_counts[lvl] = zoomed_counts.get(lvl, 0) + 1
            overlays.append({
                "x": X, "y": Y, "size": S,
                "level": lvl,
                "type": "zoomed",
                "orig_x": x, "orig_y": y,
                "score": "",
                "fill": zoomed_fill,
                "border": zoomed_border,
            })

        # -----------------------------------------------------------------
        # 5. Build HTML
        # -----------------------------------------------------------------
        level_ids = sorted(
            set(list(active_counts.keys()) + list(zoomed_counts.keys()))
        )
        if not level_ids:
            level_ids = [thumb_level]

        total_all = sum(
            sum(1 for _ in self.iterate_patches(lvl))
            for lvl in self.levels_info
            if not self.levels_info[lvl].get("frozen", False)
        )

        metadata_html = ""
        if metadata:
            rows = "".join(
                f'<div class="meta-row">'
                f'<span class="meta-key">{k}</span>'
                f'<span class="meta-val">{v}</span>'
                f'</div>'
                for k, v in metadata.items()
            )
            metadata_html = (
                f'<div class="meta-section">'
                f'<div class="meta-toggle" onclick="toggleMeta()" title="Toggle metadata">'
                f'Metadata <span id="meta-arrow">&#9654;</span>'
                f'</div>'
                f'<div class="meta-table" id="meta-body" style="display:none">{rows}</div>'
                f'</div>'
            )

        level_state_json = json.dumps(
            {lvl: {"active": True, "zoomed": True} for lvl in level_ids}
        )

        level_swatches = ""
        for lvl in level_ids:
            a_cnt = active_counts.get(lvl, 0)
            z_cnt = zoomed_counts.get(lvl, 0)
            depth = self.max_level - lvl
            level_swatches += f"""
                <div class="legend-level">
                    <div class="legend-level-header">
                        <span>Level {lvl} (depth&nbsp;{depth})</span>
                    </div>
                    <div class="legend-item">
                        <input type="checkbox" class="legend-checkbox"
                               id="toggle-active-lvl-{lvl}" checked
                               onchange="toggleLevelType({lvl}, 'active')">
                        <div class="legend-color"
                             style="background:{active_color(lvl)};"></div>
                        <label for="toggle-active-lvl-{lvl}">Active ({a_cnt})</label>
                    </div>
                    <div class="legend-item">
                        <input type="checkbox" class="legend-checkbox"
                               id="toggle-zoomed-lvl-{lvl}" checked
                               onchange="toggleLevelType({lvl}, 'zoomed')">
                        <div class="legend-color"
                             style="background:{zoomed_fill};"></div>
                        <label for="toggle-zoomed-lvl-{lvl}">Zoomed ({z_cnt})</label>
                    </div>
                </div>
"""

        overlay_html = ""
        for ov in overlays:
            tooltip = (
                f"{ov['type'].capitalize()} | Level {ov['level']} | "
                f"x={ov['orig_x']}, y={ov['orig_y']}"
                + (f" | score={ov['score']}" if ov["score"] else "")
            )
            overlay_html += f"""
                <div class="patch-overlay"
                     data-level="{ov['level']}"
                     data-type="{ov['type']}"
                     style="left:{ov['x']}px; top:{ov['y']}px;
                            width:{ov['size']}px; height:{ov['size']}px;
                            background-color:{ov['fill']};
                            border-color:{ov['border']};"
                     title="{tooltip}">
                </div>
"""

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>WSI Patch Visualization</title>
    <style>
        * {{ box-sizing: border-box; }}
        html, body {{
            margin: 0; padding: 0; height: 100%;
            background: #2a2a2a; color: #ffffff;
            font-family: monospace;
        }}
        #main {{ display: flex; height: 100%; }}
        #image-panel {{ flex: 1; overflow: auto; padding: 20px; }}
        #control-panel {{
            width: 340px; min-width: 340px;
            background: #1a1a1a; padding: 15px;
            overflow-y: auto; border-left: 2px solid #444;
        }}
        #container {{ position: relative; display: inline-block; }}
        #wsi-image {{ display: block; }}
        .patch-overlay {{
            position: absolute; box-sizing: border-box;
            pointer-events: all; opacity: 0.35;
            cursor: pointer; border-style: solid; border-width: 2px;
        }}
        .patch-overlay:hover {{ opacity: 0.85; z-index: 10; }}
        .legend {{ margin-top: 15px; padding-top: 15px; border-top: 1px solid #555; }}
        .legend-item {{
            display: flex; align-items: center;
            margin: 5px 0; cursor: pointer; user-select: none;
        }}
        .legend-item label {{ cursor: pointer; }}
        .legend-color {{
            width: 20px; height: 20px;
            margin-right: 10px; border: 1px solid #fff;
            flex-shrink: 0;
        }}
        .legend-checkbox {{
            margin-right: 10px; width: 18px; height: 18px; cursor: pointer;
        }}
        .legend-level {{
            border: 1px solid #444; border-radius: 6px;
            padding: 8px; margin-bottom: 10px;
        }}
        .legend-level-header {{
            margin-bottom: 6px; font-weight: bold;
        }}
        .legend-actions {{
            display: flex; justify-content: flex-end;
            gap: 8px; margin-bottom: 10px;
        }}
        .legend-actions button {{
            background: #444; color: #fff;
            border: 1px solid #777; border-radius: 4px;
            padding: 4px 10px; cursor: pointer;
        }}
        .legend-actions button:hover {{ background: #666; }}
        h3 {{ margin-top: 0; }}
        .stat {{ margin: 4px 0; }}
        .meta-section {{
            margin-bottom: 12px; border-bottom: 1px solid #555; padding-bottom: 10px;
        }}
        .meta-toggle {{
            cursor: pointer; user-select: none;
            color: #ccc; font-size: 0.85em;
            padding: 3px 0; display: flex; align-items: center; gap: 6px;
        }}
        .meta-toggle:hover {{ color: #fff; }}
        #meta-arrow {{ font-size: 0.75em; transition: transform 0.15s; }}
        .meta-table {{
            margin-top: 8px;
        }}
        .meta-row {{
            display: flex; justify-content: space-between;
            margin: 3px 0; gap: 8px;
        }}
        .meta-key {{ color: #aaa; white-space: nowrap; font-size: 0.85em; }}
        .meta-val {{ font-weight: bold; text-align: right; word-break: break-all; font-size: 0.85em; }}
    </style>
</head>
<body>
<div id="main">
    <div id="image-panel">
        <div id="container">
            <img id="wsi-image"
                 src="data:image/png;base64,{img_base64}"
                 width="{display_width}" height="{display_height}">
{overlay_html}
        </div>
    </div>
    <div id="control-panel">
        <h3>WSI Patch Visualization</h3>
{metadata_html}
        <div class="stat">All patches (pyramid): <strong>{total_all}</strong></div>
        <div class="stat">Active patches: <strong>{len(self.active_patches)}</strong></div>
        <div class="stat">Zoomed-past patches: <strong>{len(self.zoomed_patches)}</strong></div>
        <div class="stat">Thumbnail level: <strong>{thumb_level}</strong></div>
        <div class="stat">Thumbnail size: <strong>{thumb_w}&thinsp;&times;&thinsp;{thumb_h}</strong></div>
        <div class="legend">
            <div class="legend-actions">
                <button type="button" onclick="showAllLevels()">Show All</button>
            </div>
            <strong>Per-level visibility</strong><br>
            <small>
                green&nbsp;=&nbsp;active&nbsp;(darker&nbsp;=&nbsp;finer&nbsp;level),
                red&nbsp;=&nbsp;zoomed-past
            </small>
{level_swatches}
        </div>
    </div>
</div>

<script>
    const levelState = {level_state_json};

    function updatePatchVisibility() {{
        document.querySelectorAll('.patch-overlay').forEach(function(el) {{
            const lvl = parseInt(el.dataset.level, 10);
            const pType = el.dataset.type;
            if (!levelState[lvl]) levelState[lvl] = {{ active: true, zoomed: true }};
            el.style.display = levelState[lvl][pType] !== false ? 'block' : 'none';
        }});
    }}

    function toggleLevelType(level, patchType) {{
        if (!levelState[level]) levelState[level] = {{ active: true, zoomed: true }};
        const checkbox = document.getElementById(
            'toggle-' + patchType + '-lvl-' + level
        );
        levelState[level][patchType] = checkbox ? checkbox.checked : true;
        updatePatchVisibility();
    }}

    function showAllLevels() {{
        Object.keys(levelState).forEach(function(lk) {{
            ['active', 'zoomed'].forEach(function(t) {{
                levelState[lk][t] = true;
                const cb = document.getElementById('toggle-' + t + '-lvl-' + lk);
                if (cb) cb.checked = true;
            }});
        }});
        updatePatchVisibility();
    }}

    document.querySelectorAll('.patch-overlay').forEach(function(el) {{
        el.addEventListener('mouseenter', function() {{
            this.style.opacity = '0.85';
            this.style.zIndex = '100';
        }});
        el.addEventListener('mouseleave', function() {{
            this.style.opacity = '0.35';
            this.style.zIndex = '1';
        }});
    }});

    updatePatchVisibility();

    function toggleMeta() {{
        const body = document.getElementById('meta-body');
        const arrow = document.getElementById('meta-arrow');
        if (!body) return;
        if (body.style.display === 'none') {{
            body.style.display = 'block';
            if (arrow) arrow.innerHTML = '&#9660;';
        }} else {{
            body.style.display = 'none';
            if (arrow) arrow.innerHTML = '&#9654;';
        }}
    }}
</script>
</body>
</html>
"""

        # -----------------------------------------------------------------
        # 6. Write to disk
        # -----------------------------------------------------------------
        os.makedirs(os.path.dirname(os.path.abspath(output_html)), exist_ok=True)
        with open(output_html, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"[WSI viz] Saved \u2192 {output_html}")
        print(f"  active : {len(self.active_patches)} patches across levels "
              f"{sorted(active_counts)}")
        print(f"  zoomed : {len(self.zoomed_patches)} patches")
        return output_html
