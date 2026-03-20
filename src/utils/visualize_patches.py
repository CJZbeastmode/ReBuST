import os
import json
import base64
from io import BytesIO
from PIL import Image


def generate_visualization(
    wsi,
    all_patches_count,
    kept,  # List[Tuple[patch, meta]] where meta = {"level": int, "x": int, "y": int}
    zoomed=None,  # List[Tuple[patch, meta]] where meta = {"level": int, "x": int, "y": int}
    output_html="./data/visualization.html",
):
    """
    Render an HTML overlay of kept/zoomed patches on the WSI thumbnail.
    
    Parameters
    ----------
    wsi : WSI
        WSI object with slide and level information
    all_patches_count : int
        Total number of patches considered
    kept : List[Tuple[patch, dict]]
        List of (patch, metadata) tuples for patches that were kept.
        Each metadata dict should contain: {"level": int, "x": int, "y": int}
    zoomed : List[Tuple[patch, dict]], optional
        List of (patch, metadata) tuples for patches that were zoomed past.
        Each metadata dict should contain: {"level": int, "x": int, "y": int}
    output_html : str
        Path to save the HTML visualization file
        
    Returns
    -------
    str
        Path to the saved HTML file
    """

    if zoomed is None:
        zoomed = []

    # -------------------------------------------------------------------------
    # 1. Load thumbnail at max_level (coarsest level)
    # -------------------------------------------------------------------------
    thumb_level = wsi.max_level
    thumb_w, thumb_h = wsi.levels_info[thumb_level]["size"]

    # One pixel in the thumbnail is one pixel at thumb_level
    thumbnail = wsi.slide.read_region((0, 0), thumb_level, (thumb_w, thumb_h))
    thumbnail = thumbnail.convert("RGB")

    # Encode thumbnail as base64 for inline HTML
    buf = BytesIO()
    thumbnail.save(buf, format="PNG")
    img_base64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # Display scale (you asked for 1.0)
    display_scale = 1.0
    display_width = int(thumb_w * display_scale)
    display_height = int(thumb_h * display_scale)

    # -------------------------------------------------------------------------
    # 2. Coordinate mapping: (level, x, y, patch_size) → thumbnail coords
    #
    # Core idea:
    #   Define a "base" reference (level 0) using its width:
    #   ds[l] = width_level0 / width_level_l
    #
    #   To map from some level L to thumbnail level T:
    #       x_T = x_L * ds[L] / ds[T]
    #       size_T = patch_size * ds[L] / ds[T]
    #
    # Because thumbnail is just level T drawn 1:1, x_T, size_T are in thumbnail pixels.
    # -------------------------------------------------------------------------
    level0_w, _ = wsi.levels_info[0]["size"]

    # Precompute downsample factors relative to level 0 for all levels
    ds = {}
    for lvl, info in wsi.levels_info.items():
        w_l, _ = info["size"]
        if w_l == 0:
            ds[lvl] = 1.0
        else:
            ds[lvl] = level0_w / float(w_l)

    ds_thumb = ds[thumb_level]

    def to_thumb(lvl, x, y, patch_size):
        """Map (lvl, x, y, patch_size) in level coordinates to thumbnail coords (x, y, size)."""
        ds_src = ds[lvl]

        # Map to base level 0 coords, then down to thumbnail level (thumb_level)
        # x0 = x * ds_src
        # x_thumb = x0 / ds_thumb = x * ds_src / ds_thumb
        x_thumb = x * ds_src / ds_thumb
        y_thumb = y * ds_src / ds_thumb

        size_thumb = patch_size * ds_src / ds_thumb

        # Apply display scale and clamp to at least 1 pixel
        X = int(x_thumb * display_scale)
        Y = int(y_thumb * display_scale)
        S = max(1, int(size_thumb * display_scale))

        return X, Y, S

    # -------------------------------------------------------------------------
    # 3. Build overlay metadata
    # -------------------------------------------------------------------------
    overlays = []

    # Colors per level for kept patches
    kept_color = "#00FF00"
    zoomed_color = "#FF0000"

    kept_counts = {}
    zoomed_counts = {}

    # Kept patches
    for lvl, x, y in kept:

        X, Y, S = to_thumb(lvl, x, y, wsi.patch_size)

        kept_counts[lvl] = kept_counts.get(lvl, 0) + 1
        overlays.append(
            {
                "x": X,
                "y": Y,
                "size": S,
                "level": lvl,
                "type": "kept",
                "orig_x": x,
                "orig_y": y,
            }
        )

    # Zoomed patches
    for lvl, x, y in zoomed:

        X, Y, S = to_thumb(lvl, x, y, wsi.patch_size)

        zoomed_counts[lvl] = zoomed_counts.get(lvl, 0) + 1
        overlays.append(
            {
                "x": X,
                "y": Y,
                "size": S,
                "level": lvl,
                "type": "zoomed",
                "orig_x": x,
                "orig_y": y,
            }
        )

    # -------------------------------------------------------------------------
    # 4. Build HTML
    # -------------------------------------------------------------------------
    level_ids = sorted(set(list(kept_counts.keys()) + list(zoomed_counts.keys())))
    if not level_ids:
        level_ids = [thumb_level]

    level_state_json = json.dumps(
        {lvl: {"kept": True, "zoomed": True} for lvl in level_ids}
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>WSI Patch Visualization</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        html, body {{
            margin: 0;
            padding: 0;
            height: 100%;
            background: #2a2a2a;
            color: #ffffff;
            font-family: monospace;
        }}
        #main {{
            display: flex;
            height: 100%;
        }}
        #image-panel {{
            flex: 1;
            overflow: auto;
            padding: 20px;
        }}
        #control-panel {{
            width: 340px;
            min-width: 340px;
            background: #1a1a1a;
            padding: 15px;
            overflow-y: auto;
            border-left: 2px solid #444;
        }}
        #container {{
            position: relative;
            display: inline-block;
        }}
        #wsi-image {{
            display: block;
        }}
        .patch-overlay {{
            position: absolute;
            box-sizing: border-box;
            pointer-events: all;
            opacity: 0.3;
            cursor: pointer;
            border-style: solid;
            border-width: 2px;
        }}
        .patch-overlay:hover {{
            opacity: 0.8;
            z-index: 10;
        }}
        .legend {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #555;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
            cursor: pointer;
            user-select: none;
        }}
        .legend-item label {{
            cursor: pointer;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            margin-right: 10px;
            border: 1px solid #fff;
        }}
        .legend-checkbox {{
            margin-right: 10px;
            width: 18px;
            height: 18px;
            cursor: pointer;
        }}
        .legend-level {{
            border: 1px solid #444;
            border-radius: 6px;
            padding: 8px;
            margin-bottom: 10px;
        }}
        .legend-level-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
            font-weight: bold;
        }}
        .legend-actions {{
            display: flex;
            justify-content: flex-end;
            gap: 8px;
            margin-bottom: 10px;
        }}
        .legend-level-header button,
        .legend-actions button {{
            background: #444;
            color: #fff;
            border: 1px solid #777;
            border-radius: 4px;
            padding: 4px 10px;
            cursor: pointer;
        }}
        .legend-level-header button:hover,
        .legend-actions button:hover {{
            background: #666;
        }}
        h3 {{
            margin-top: 0;
        }}
    </style>
</head>
<body>
    <div id="main">
        <div id="image-panel">
            <div id="container">
                <img id="wsi-image"
                     src="data:image/png;base64,{img_base64}"
                     width="{display_width}"
                     height="{display_height}">
"""

    # Overlays go inside the container
    for ov in overlays:
        if ov["type"] == "zoomed":
            fill = zoomed_color
            border = "#660000"
        else:
            fill = kept_color
            border = "#006600"

        html += f"""
                <div class="patch-overlay"
                     data-level="{ov['level']}"
                     data-type="{ov['type']}"
                     style="left: {ov['x']}px;
                            top: {ov['y']}px;
                            width: {ov['size']}px;
                            height: {ov['size']}px;
                            background-color: {fill};
                            border-color: {border};"
                     title="{ov['type'].capitalize()} | Level {ov['level']} | Position: x={ov['orig_x']}, y={ov['orig_y']}">
                </div>
"""

    # Close image container and panel, then add control panel
    html += f"""
            </div>
        </div>
        <div id="control-panel">
            <h3>WSI Patch Visualization</h3>
            <div>All patches: <strong>{all_patches_count}</strong></div>
            <div>Kept patches: <strong>{len(kept)}</strong></div>
            <div>Zoomed patches: <strong>{all_patches_count - len(kept)}</strong></div>
            <div>Thumbnail level: <strong>{thumb_level}</strong></div>
            <div>Thumbnail size: <strong>{thumb_w} × {thumb_h}</strong></div>
            <div class="legend">
                <div class="legend-actions">
                    <button type="button" onclick="showAllLevels()">Show All</button>
                </div>
                <strong>Per-level visibility</strong><br>
                <small>(green = kept, red = zoomed)</small>
"""

    for lvl in level_ids:
        kept_cnt = kept_counts.get(lvl, 0)
        zoomed_cnt = zoomed_counts.get(lvl, 0)
        html += f"""
                <div class="legend-level">
                    <div class="legend-level-header">
                        <span>Level {lvl}</span>
                        <button type="button" onclick="soloLevelView({lvl})">Solo</button>
                    </div>
                    <div class="legend-item">
                        <input type="checkbox" class="legend-checkbox" id="toggle-kept-lvl-{lvl}" checked onchange="toggleLevelType({lvl}, 'kept')">
                        <div class="legend-color" style="background: {kept_color};"></div>
                        <label for="toggle-kept-lvl-{lvl}">Kept ({kept_cnt})</label>
                    </div>
                    <div class="legend-item">
                        <input type="checkbox" class="legend-checkbox" id="toggle-zoomed-lvl-{lvl}" checked onchange="toggleLevelType({lvl}, 'zoomed')">
                        <div class="legend-color" style="background: {zoomed_color};"></div>
                        <label for="toggle-zoomed-lvl-{lvl}">Zoomed ({zoomed_cnt})</label>
                    </div>
                </div>
"""

    html += f"""
            </div>
        </div>
    </div>

    <script>
        const levelState = {level_state_json};
        let soloLevel = null;

        function updatePatchVisibility() {{
            const overlays = document.querySelectorAll('.patch-overlay');
            overlays.forEach(function(overlay) {{
                const lvl = parseInt(overlay.dataset.level, 10);
                const patchType = overlay.dataset.type;
                if (!levelState[lvl]) levelState[lvl] = {{ kept: true, zoomed: true }};
                const typeVisible = levelState[lvl][patchType] !== false;
                const soloVisible = soloLevel === null || soloLevel === lvl;
                overlay.style.display = typeVisible && soloVisible ? 'block' : 'none';
            }});
        }}

        function toggleLevelType(level, patchType) {{
            if (!levelState[level]) levelState[level] = {{ kept: true, zoomed: true }};
            const checkbox = document.getElementById('toggle-' + patchType + '-lvl-' + level);
            levelState[level][patchType] = checkbox ? checkbox.checked : true;
            updatePatchVisibility();
        }}

        function soloLevelView(level) {{
            soloLevel = soloLevel === level ? null : level;
            updatePatchVisibility();
        }}

        function showAllLevels() {{
            soloLevel = null;
            Object.keys(levelState).forEach(function(levelKey) {{
                ['kept', 'zoomed'].forEach(function(type) {{
                    levelState[levelKey][type] = true;
                    const checkbox = document.getElementById('toggle-' + type + '-lvl-' + levelKey);
                    if (checkbox) checkbox.checked = true;
                }});
            }});
            updatePatchVisibility();
        }}

        document.querySelectorAll('.patch-overlay').forEach(function(el) {{
            el.addEventListener('mouseenter', function() {{
                this.style.opacity = '0.8';
                this.style.zIndex = '100';
            }});
            el.addEventListener('mouseleave', function() {{
                this.style.opacity = '0.3';
                this.style.zIndex = '1';
            }});
        }});

        updatePatchVisibility();
    </script>
</body>
</html>
"""

    # -------------------------------------------------------------------------
    # 5. Write HTML to disk
    # -------------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    with open(output_html, "w") as f:
        f.write(html)

    print(f"Visualization saved to: {output_html}")
    return output_html
