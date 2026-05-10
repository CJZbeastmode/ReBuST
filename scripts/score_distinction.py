"""
centroid_score_viz.py

Compute any PatchScoreModule score for every patch of a WSI and generate
an HTML that shows the score prominently above each patch.

Supported score types (--score-type):
    centroid        CancerTypeCentroidScore  (requires cancer type)
    contrastive     ContrastiveTextScore     (requires cancer type)
    imgsim          ImgSimScore
    textalign       TextAlignScore
    tissue          TissuePresenceScore
    tissue_penalty  TissuePresencePenalty
    entropy         EntropyScore

When a cancer type is required and --cancer-type is not given, it is
automatically extracted from the TCGA case name in the image filename
(e.g.  TCGA-18-4083-LUSC.svs  →  LUSC).

Usage:
    python scripts/centroid_score_viz.py
    python scripts/centroid_score_viz.py --score-type entropy --image data/images/TCGA-05-4390-LUAD.svs
    python scripts/centroid_score_viz.py --score-type centroid --cancer-type LUAD --image data/images/TCGA-05-4390-LUAD.svs
    python scripts/centroid_score_viz.py --score-type tissue --image data/images/TCGA-18-3406-LUSC.svs
"""

import sys
import argparse
import base64
from io import BytesIO
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import os
import json
import numpy as np
from PIL import Image

from src.utils.wsi import WSI
from src.utils.patch_scores import (
    CancerTypeCentroidScore,
    ContrastiveTextScore,
    ImgSimScore,
    TextAlignScore,
    TissuePresenceScore,
    TissuePresencePenalty,
    EntropyScore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def score_to_color(score: float, vmin: float, vmax: float) -> str:
    """Map a score in [vmin, vmax] to a hex colour (green=high, red=low)."""
    span = vmax - vmin if vmax > vmin else 1.0
    t = max(0.0, min(1.0, (score - vmin) / span))  # 0 = low, 1 = high
    r = int(220 * (1.0 - t))
    g = int(200 * t)
    b = 40
    return f"#{r:02X}{g:02X}{b:02X}"


def patch_to_b64(patch) -> str:
    """Encode a PIL patch as a base64 PNG data-URI."""
    buf = BytesIO()
    patch.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def thumb_to_b64(img) -> str:
    """Encode WSI thumbnail as base64 PNG."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def extract_cancer_type_from_path(image_path: str) -> str:
    """
    Extract cancer type from a TCGA case filename.
    E.g. 'TCGA-18-4083-LUSC.svs' → 'LUSC'
    Falls back to an empty string when the pattern is not recognised.
    """
    stem = Path(image_path).stem  # e.g. 'TCGA-18-4083-LUSC'
    parts = stem.split("-")
    if len(parts) >= 4 and parts[3].isalpha():
        return parts[3].upper()
    # Last segment as a fallback (handles e.g. 'my_slide-LUAD')
    last = parts[-1]
    if last.isalpha() and 2 <= len(last) <= 6:
        return last.upper()
    return ""


_SCORE_TYPE_ALIASES = {
    "centroid": "centroid",
    "contrastive": "contrastive",
    "imgsim": "imgsim",
    "textalign": "textalign",
    "tissue": "tissue",
    "tissue_penalty": "tissue_penalty",
    "entropy": "entropy",
}


def build_scorer(score_type: str, cancer_type: str, embedder, neg_types: str = "all"):
    """
    Factory: return a configured PatchScoreModule for *score_type*.

    Parameters
    ----------
    score_type:
        One of: centroid, contrastive, imgsim, textalign,
                tissue, tissue_penalty, entropy.
    cancer_type:
        Required for 'centroid' and 'contrastive'; ignored otherwise.
    embedder:
        Shared PLIP/CONCH embedder (already loaded by WSI).
    neg_types:
        For 'contrastive' only. One of:
          'all'   – all other available cancer types (default)
          'pairs' – biologically curated pairs from CONTRASTIVE_PAIRS
          'a,b,..'– explicit comma-separated cancer type codes
    """
    st = score_type.lower()
    if st == "centroid":
        if not cancer_type:
            raise ValueError(
                "--score-type centroid requires a cancer type. "
                "Pass --cancer-type or use a TCGA filename."
            )
        scorer = CancerTypeCentroidScore(
            cancer_type=cancer_type,
            weight=1.0,
            embedder=embedder,
        )
        # Pre-build centroid so it shows up in the log
        _ = scorer._get_centroid()
        return scorer

    elif st == "contrastive":
        if not cancer_type:
            raise ValueError(
                "--score-type contrastive requires a cancer type. "
                "Pass --cancer-type or use a TCGA filename."
            )
        # Parse neg_types: 'all', 'pairs', or comma-separated list
        if neg_types in ("all", "pairs"):
            resolved_neg = neg_types
        else:
            resolved_neg = [
                t.strip().upper() for t in neg_types.split(",") if t.strip()
            ]
        scorer = ContrastiveTextScore(
            pos_cancer_type=cancer_type,
            neg_cancer_types=resolved_neg,
            weight=1.0,
            embedder=embedder,
        )
        return scorer

    elif st == "imgsim":
        return ImgSimScore(weight=10.0, embedder=embedder)

    elif st == "textalign":
        return TextAlignScore(weight=1.0, embedder=embedder)

    elif st == "tissue":
        return TissuePresenceScore(weight=1.0)

    elif st == "tissue_penalty":
        return TissuePresencePenalty(weight=1.0)

    elif st == "entropy":
        return EntropyScore(weight=1.0)

    else:
        raise ValueError(
            f"Unknown score_type '{score_type}'. "
            f"Choose from: {', '.join(_SCORE_TYPE_ALIASES)}."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    image_path: str,
    score_type: str,
    output_html: str,
    neg_types: str = "all",  # "all" | "pairs" | comma-separated list
    max_patches: int | None = None,  # per-level cap
):
    print(f"[INFO] Loading WSI: {image_path}")
    wsi = WSI(image_path)

    print(
        f"[INFO] max_level={wsi.max_level}  min_level={wsi.min_level}  "
        f"patch_size={wsi.patch_size}"
    )

    # All non-frozen levels, coarsest → finest
    all_levels = sorted(
        [lvl for lvl, info in wsi.levels_info.items() if not info.get("frozen", False)],
        reverse=True,
    )
    for lvl in all_levels:
        n = sum(1 for _ in wsi.iterate_patches(lvl))
        capped = (
            f"  (capped at {max_patches})" if max_patches and n > max_patches else ""
        )
        print(
            f"[INFO]   level {lvl:2d}  type={wsi.levels_info[lvl]['type']:9s}  "
            f"size={str(wsi.levels_info[lvl]['size']):18s}  patches={n}{capped}"
        )

    # ------------------------------------------------------------------
    # Resolve cancer type (auto-detect from TCGA filename if not supplied)
    # ------------------------------------------------------------------
    resolved_cancer_type = extract_cancer_type_from_path(image_path)
    if resolved_cancer_type:
        print(f"[INFO] Cancer type: {resolved_cancer_type}")
    elif score_type in ("centroid", "contrastive"):
        raise ValueError(
            f"--score-type {score_type} requires a cancer type but none could be determined. "
        )

    # ------------------------------------------------------------------
    # Build scorer
    # ------------------------------------------------------------------
    print(f"[INFO] Building scorer: {score_type}  neg_types={neg_types} ...")
    scorer = build_scorer(
        score_type, resolved_cancer_type, wsi.embedder, neg_types=neg_types
    )
    print(f"[INFO] Scorer ready: {scorer.__class__.__name__}")

    # ------------------------------------------------------------------
    # Score every patch at every non-frozen level
    # ------------------------------------------------------------------
    results = []  # list of {key, b64, score}
    results_by_level: dict[int, list] = {}  # lvl → list of result dicts

    for lvl in all_levels:
        coords = list(wsi.iterate_patches(lvl))
        if max_patches:
            coords = coords[:max_patches]
        level_results = []
        total = len(coords)
        print(f"[SCORE] level {lvl} — {total} patches ...")
        for i, (x, y) in enumerate(coords):
            if i % 50 == 0 and i > 0:
                print(f"[SCORE]   {i}/{total}")
            try:
                patch = wsi.get_patch(lvl, x, y)
                score = scorer.compute_stop(parent_patch=patch)
            except Exception as e:
                print(f"[WARN] patch ({lvl},{x},{y}): {e}")
                patch = Image.new(
                    "RGB", (wsi.patch_size, wsi.patch_size), (200, 200, 200)
                )
                score = 0.0

            patch_b64 = patch_to_b64(patch)
            rec = {"key": (lvl, x, y), "b64": patch_b64, "score": score}
            results.append(rec)
            level_results.append(rec)

        results_by_level[lvl] = level_results

        lscores = [r["score"] for r in level_results]
        if lscores:
            print(
                f"[SCORE]   level {lvl} done — "
                f"min={min(lscores):.4f}  max={max(lscores):.4f}  "
                f"mean={float(np.mean(lscores)):.4f}"
            )

    scores = [r["score"] for r in results]
    s_min, s_max = float(min(scores)), float(max(scores))
    s_mean = float(np.mean(scores))
    print(
        f"[INFO] Overall score stats — "
        f"min={s_min:.4f}  max={s_max:.4f}  mean={s_mean:.4f}"
    )

    # ------------------------------------------------------------------
    # WSI thumbnail (for the overview panel)
    # ------------------------------------------------------------------
    thumb_level = wsi.max_level
    level_entry = wsi.levels_info[thumb_level]
    if level_entry.get("type", "native").lower() == "synthetic":
        thumbnail = wsi.synthetic_images[thumb_level].convert("RGB")
    else:
        tw, th = level_entry["size"]
        thumbnail = wsi.slide.read_region(
            (0, 0), level_entry["native_idx"], (tw, th)
        ).convert("RGB")

    thumb_b64 = thumb_to_b64(thumbnail)
    thumb_w, thumb_h = thumbnail.size

    # Build overview overlays (one rect per root patch, coloured by score)
    level0_w, _ = wsi.levels_info[0]["size"]
    ds: dict[int, float] = {}
    for lvl2, info in wsi.levels_info.items():
        w_l, _ = info["size"]
        ds[lvl2] = level0_w / float(w_l) if w_l > 0 else 1.0
    ds_thumb = ds[thumb_level]

    overview_overlays = []
    for r in results:
        lvl, x, y = r["key"]
        ratio = ds[lvl] / ds_thumb
        OX = int(x * ratio)
        OY = int(y * ratio)
        OS = max(1, int(wsi.patch_size * ratio))
        col = score_to_color(r["score"], s_min, s_max)
        overview_overlays.append(
            {
                "x": OX,
                "y": OY,
                "size": OS,
                "score": f"{r['score']:.4f}",
                "color": col,
            }
        )

    # ------------------------------------------------------------------
    # Render patch cards HTML — grouped by level
    # ------------------------------------------------------------------
    cards_html_parts = []
    for lvl in all_levels:
        lvl_results = results_by_level.get(lvl, [])
        if not lvl_results:
            continue
        lscores = [r["score"] for r in lvl_results]
        l_min, l_max, l_mean = min(lscores), max(lscores), float(np.mean(lscores))
        info = wsi.levels_info[lvl]
        level_type = info.get("type", "native")
        w, h = info["size"]
        n_cols = max(1, w // wsi.patch_size)
        n_rows = max(1, h // wsi.patch_size)

        cards_html_parts.append(
            f'<div class="level-header" id="level-{lvl}">'
            f" Level {lvl}"
            f" <small>({level_type} &nbsp;|&nbsp; {w}&thinsp;&times;&thinsp;{h}px"
            f" &nbsp;|&nbsp; {n_cols}&thinsp;&times;&thinsp;{n_rows} patches"
            f" &nbsp;|&nbsp; min={l_min:.4f} &nbsp; max={l_max:.4f} &nbsp; mean={l_mean:.4f})</small>"
            f"</div>"
        )

        for r in lvl_results:
            l, x, y = r["key"]
            sc = r["score"]
            col = score_to_color(sc, s_min, s_max)
            brightness = (
                int(col[1:3], 16) * 0.299
                + int(col[3:5], 16) * 0.587
                + int(col[5:7], 16) * 0.114
            )
            txt_col = "#000" if brightness > 128 else "#fff"
            cards_html_parts.append(
                f"""
        <div class="card" data-level="{lvl}">
            <div class="score-badge" style="background:{col}; color:{txt_col};">
                {sc:.4f}
            </div>
            <img class="patch-img" src="{r['b64']}" loading="lazy"
                 title="lvl={l} x={x} y={y}  score={sc:.6f}">
            <div class="card-footer">lvl={l}&nbsp; x={x}&nbsp; y={y}</div>
        </div>"""
            )

    cards_html = "\n".join(cards_html_parts)

    # Per-level sidebar stats HTML
    level_stats_html = ""
    for lvl in all_levels:
        lvl_results = results_by_level.get(lvl, [])
        if not lvl_results:
            continue
        lscores = [r["score"] for r in lvl_results]
        level_type = wsi.levels_info[lvl].get("type", "native")
        level_stats_html += (
            f'<div class="lvl-stat">'
            f'<span class="lvl-label">Level {lvl} <small>({level_type})</small></span>'
            f'<span class="lvl-nums">'
            f"n={len(lvl_results)} &nbsp; "
            f"min={min(lscores):.3f} &nbsp; "
            f"max={max(lscores):.3f} &nbsp; "
            f"mean={float(np.mean(lscores)):.3f}"
            f"</span>"
            f'<a class="lvl-jump" href="#level-{lvl}">↓ jump</a>'
            f"</div>"
        )

    # Level filter checkboxes for the controls bar
    level_filter_checks = " ".join(
        f'<label style="white-space:nowrap;">'
        f'<input type="checkbox" checked onchange="filterLevel({lvl}, this.checked)">'
        f" L{lvl}</label>"
        for lvl in all_levels
    )

    # Overview overlay JS data
    ov_json = json.dumps(overview_overlays)

    # ------------------------------------------------------------------
    # Assemble full HTML
    # ------------------------------------------------------------------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{scorer.__class__.__name__} – {resolved_cancer_type or Path(image_path).stem} – {Path(image_path).name}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    background: #1a1a1a;
    color: #eee;
    font-family: monospace;
    overflow-x: hidden;
}}

/* ---- Top bar ---- */
#topbar {{
    position: sticky; top: 0; z-index: 100;
    background: #111;
    border-bottom: 2px solid #444;
    display: flex; align-items: center; gap: 20px;
    padding: 10px 20px;
}}
#topbar h1 {{ font-size: 1.1em; white-space: nowrap; }}
.stat-pill {{
    background: #2a2a2a; border: 1px solid #555;
    border-radius: 4px; padding: 3px 10px;
    font-size: 0.82em; white-space: nowrap;
}}
.stat-pill span {{ font-weight: bold; color: #9df; }}

/* ---- Layout ---- */
#layout {{ display: flex; height: calc(100vh - 52px); }}

/* ---- Sidebar ---- */
#sidebar {{
    width: 320px; min-width: 320px;
    background: #141414;
    border-right: 2px solid #333;
    overflow-y: auto;
    padding: 14px;
}}
#sidebar h2 {{ font-size: 0.95em; margin-bottom: 8px; color: #aaa; }}
#overview-wrap {{ position: relative; display: inline-block; }}
#overview-img {{ max-width: 290px; display: block; }}
.ov-rect {{
    position: absolute;
    border: 1.5px solid rgba(255,255,255,0.45);
    opacity: 0.65;
    cursor: pointer;
    transition: opacity 0.1s;
}}
.ov-rect:hover {{ opacity: 1.0; z-index: 5; }}

/* Colour scale legend */
#legend {{
    margin-top: 14px;
    font-size: 0.78em;
    color: #aaa;
}}
#grad-bar {{
    width: 100%; height: 14px; border-radius: 3px;
    background: linear-gradient(to right, #DC0028, #a03000, #00AA28);
    margin: 4px 0;
}}
.legend-labels {{ display: flex; justify-content: space-between; }}

/* ---- Main scroll panel ---- */
#main {{
    flex: 1;
    overflow-y: auto;
    padding: 16px;
}}

/* ---- Sort / filter controls ---- */
#controls {{
    display: flex; align-items: center; gap: 14px;
    margin-bottom: 14px; flex-wrap: wrap;
}}
#controls label {{ font-size: 0.85em; color: #bbb; }}
select, input[type=range] {{ background: #2a2a2a; color: #eee;
    border: 1px solid #555; border-radius: 3px; padding: 3px 6px; }}

/* ---- Grid ---- */
#grid {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}}

/* ---- Patch card ---- */
.card {{
    width: 180px;
    background: #242424;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    align-items: center;
    transition: transform 0.1s;
}}
.card:hover {{ transform: scale(1.04); z-index: 5; border-color: #888; }}

.score-badge {{
    width: 100%;
    text-align: center;
    font-size: 1.55em;
    font-weight: bold;
    padding: 6px 4px;
    letter-spacing: 0.04em;
}}

.patch-img {{
    width: 100%;
    display: block;
    image-rendering: pixelated;
}}

.card-footer {{
    font-size: 0.68em;
    color: #888;
    padding: 4px 6px;
    width: 100%;
    text-align: center;
}}

/* ---- Level group header ---- */
.level-header {{
    width: 100%;
    padding: 8px 12px;
    margin: 18px 0 8px 0;
    background: #2a2a3a;
    border-left: 4px solid #66aaff;
    font-size: 1.0em;
    font-weight: bold;
    color: #cce;
    border-radius: 0 4px 4px 0;
}}
.level-header small {{ font-size: 0.72em; font-weight: normal; color: #99a; margin-left: 10px; }}

/* ---- Per-level sidebar stats ---- */
.lvl-stat {{
    padding: 5px 0; border-bottom: 1px solid #2a2a2a;
    font-size: 0.78em;
}}
.lvl-label {{ font-weight: bold; color: #9df; }}
.lvl-label small {{ font-weight: normal; color: #777; }}
.lvl-nums {{ display: block; color: #aaa; margin-top: 2px; }}
.lvl-jump {{ float: right; color: #66f; text-decoration: none; font-size: 0.85em; }}
.lvl-jump:hover {{ color: #aaf; }}
    border: 2px solid #fff;
    transform: scale(1.06);
}}
</style>
</head>
<body>

<!-- ===== Top bar ===== -->
<div id="topbar">
    <h1>{scorer.__class__.__name__} &mdash; <em>{resolved_cancer_type or Path(image_path).stem}</em></h1>
    <div class="stat-pill">Image: <span>{Path(image_path).name}</span></div>
    <div class="stat-pill">Total patches: <span>{len(results)}</span></div>
    <div class="stat-pill">Levels: <span>{len(all_levels)} ({wsi.min_level}–{wsi.max_level})</span></div>
    <div class="stat-pill">Min: <span>{s_min:.4f}</span></div>
    <div class="stat-pill">Max: <span>{s_max:.4f}</span></div>
    <div class="stat-pill">Mean: <span>{s_mean:.4f}</span></div>
</div>

<!-- ===== Layout ===== -->
<div id="layout">

    <!-- Sidebar: WSI overview + legend -->
    <div id="sidebar">
        <h2>WSI overview (max_level={thumb_level})</h2>
        <div id="overview-wrap">
            <img id="overview-img" src="{thumb_b64}">
            <!-- coloured rects injected by JS -->
        </div>
        <div id="legend">
            <div id="grad-bar"></div>
            <div class="legend-labels">
                <span>Low ({s_min:.4f})</span>
                <span>High ({s_max:.4f})</span>
            </div>
        </div>
        <div style="margin-top:16px;">
            <h2>Per-level stats</h2>
            {level_stats_html}
        </div>
    </div>

    <!-- Main: patch grid -->
    <div id="main">
        <div id="controls">
            <label>Sort:
                <select id="sort-select" onchange="applySort()">
                    <option value="default">Default (level/grid order)</option>
                    <option value="desc">Score ↓ high → low</option>
                    <option value="asc">Score ↑ low → high</option>
                </select>
            </label>
            <label>Min score:
                <input type="range" id="filter-min"
                       min="{s_min:.6f}" max="{s_max:.6f}"
                       step="{max(0.0001, (s_max - s_min) / 1000):.6f}"
                       value="{s_min:.6f}" oninput="applyFilter()">
                <span id="filter-val">{s_min:.4f}</span>
            </label>
            <span style="font-size:0.82em;color:#aaa;">Levels:&nbsp;{level_filter_checks}</span>
        </div>
        <div id="grid">
{cards_html}
        </div>
    </div>
</div>

<script>
// -----------------------------------------------------------------------
// 1.  Overview coloured rects
// -----------------------------------------------------------------------
const OV_DATA = {ov_json};

const ovImg  = document.getElementById('overview-img');
const ovWrap = document.getElementById('overview-wrap');

function buildOverlay() {{
    // Remove old rects
    ovWrap.querySelectorAll('.ov-rect').forEach(el => el.remove());

    const naturalW = {thumb_w};
    const naturalH = {thumb_h};
    const dispW    = ovImg.offsetWidth  || ovImg.naturalWidth;
    const dispH    = ovImg.offsetHeight || ovImg.naturalHeight;
    const scaleX   = dispW  / naturalW;
    const scaleY   = dispH  / naturalH;

    OV_DATA.forEach((ov, idx) => {{
        const el = document.createElement('div');
        el.className = 'ov-rect';
        el.style.left   = (ov.x * scaleX) + 'px';
        el.style.top    = (ov.y * scaleY) + 'px';
        el.style.width  = (ov.size * scaleX) + 'px';
        el.style.height = (ov.size * scaleY) + 'px';
        el.style.background = ov.color;
        el.title = 'score = ' + ov.score;
        el.dataset.idx = idx;
        el.addEventListener('click', () => highlightCard(idx));
        ovWrap.appendChild(el);
    }});
}}

ovImg.addEventListener('load', buildOverlay);
window.addEventListener('resize', buildOverlay);
if (ovImg.complete) buildOverlay();

// -----------------------------------------------------------------------
// 2.  Highlight card from overview click
// -----------------------------------------------------------------------
function highlightCard(idx) {{
    document.querySelectorAll('.card.highlighted')
            .forEach(el => el.classList.remove('highlighted'));
    const cards = document.querySelectorAll('#grid .card');
    if (idx < cards.length) {{
        cards[idx].classList.add('highlighted');
        cards[idx].scrollIntoView({{behavior: 'smooth', block: 'center'}});
    }}
}}

// -----------------------------------------------------------------------
// 3.  Sort / filter
// -----------------------------------------------------------------------
const SCORES = [{", ".join(f"{r['score']:.8f}" for r in results)}];

// Tag each card with its original index and score for JS use
(function() {{
    const cards = document.querySelectorAll('#grid .card');
    cards.forEach((c, i) => {{
        c.dataset.origIdx = i;
        c.dataset.score   = SCORES[i];
    }});
}})();

function applySort() {{
    const mode  = document.getElementById('sort-select').value;
    const grid  = document.getElementById('grid');
    const headers = Array.from(grid.querySelectorAll('.level-header'));
    const cards   = Array.from(grid.querySelectorAll('.card'));

    if (mode === 'default') {{
        // Restore everything to original DOM order (headers interleaved)
        const all = Array.from(grid.children);
        all.sort((a, b) => {{
            const ia = parseInt(a.dataset.origIdx ?? a.dataset.headerIdx ?? -1);
            const ib = parseInt(b.dataset.origIdx ?? b.dataset.headerIdx ?? -1);
            return ia - ib;
        }});
        headers.forEach(h => h.style.display = '');
        all.forEach(el => grid.appendChild(el));
    }} else {{
        // Score sort: hide headers, sort only cards
        headers.forEach(h => h.style.display = 'none');
        if (mode === 'desc') {{
            cards.sort((a, b) => parseFloat(b.dataset.score) - parseFloat(a.dataset.score));
        }} else {{
            cards.sort((a, b) => parseFloat(a.dataset.score) - parseFloat(b.dataset.score));
        }}
        cards.forEach(c => grid.appendChild(c));
    }}
}}

// Give headers index anchors for default-sort restore
(function() {{
    const grid = document.getElementById('grid');
    Array.from(grid.querySelectorAll('.level-header')).forEach((h, i) => {{
        h.dataset.headerIdx = -(i + 1);  // negative so they sort before cards in default
    }});
}})();

function applyFilter() {{
    const minVal = parseFloat(document.getElementById('filter-min').value);
    document.getElementById('filter-val').textContent = minVal.toFixed(4);
    document.querySelectorAll('#grid .card').forEach(c => {{
        const sc = parseFloat(c.dataset.score);
        const lvShow = c.dataset.levelVisible !== 'false';
        c.style.display = (sc >= minVal && lvShow) ? '' : 'none';
    }});
}}

// -----------------------------------------------------------------------
// 4.  Level toggle
// -----------------------------------------------------------------------
function filterLevel(lvl, show) {{
    const grid = document.getElementById('grid');
    // toggle cards for this level
    grid.querySelectorAll(`.card[data-level="${{lvl}}"]`).forEach(c => {{
        c.dataset.levelVisible = show ? 'true' : 'false';
        const minVal = parseFloat(document.getElementById('filter-min').value);
        const sc = parseFloat(c.dataset.score);
        c.style.display = (show && sc >= minVal) ? '' : 'none';
    }});
    // toggle section header
    const hdr = document.getElementById('level-' + lvl);
    if (hdr) hdr.style.display = show ? '' : 'none';
}}
</script>
</body>
</html>"""

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(output_html)), exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as fh:
        fh.write(html)

    print(f"[DONE] Saved → {output_html}")
    return output_html


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise any PatchScoreModule score for all patches of a WSI."
    )
    parser.add_argument(
        "--image",
        type=str,
        default="data/images/TCGA-18-4083-LUSC.svs",
        help="Path to the .svs file",
    )
    parser.add_argument(
        "--score-type",
        type=str,
        default="imgsim",
        choices=[
            "centroid",
            "contrastive",
            "imgsim",
            "textalign",
            "tissue",
            "tissue_penalty",
            "entropy",
        ],
        help="Score type to visualise (default: contrastive)",
    )
    parser.add_argument(
        "--neg-types",
        type=str,
        default="all",
        help="Negatives for contrastive scorer: 'all' (default), 'pairs' "
        "(biologically curated CONTRASTIVE_PAIRS), or comma-separated "
        "list e.g. 'LUSC,COAD'.",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Limit to first N patches *per level* (for quick testing)",
    )
    args = parser.parse_args()

    run(
        image_path=args.image,
        score_type=args.score_type,
        neg_types=args.neg_types,
        output_html=f"data/visualizations/score_distinction/{args.score_type}.html",
        max_patches=args.max_patches,
    )
