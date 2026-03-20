# A2C Inference Pipelines (Level 1-4)

This directory contains inference scripts for all 4 progressive A2C models.

## Overview

Each level has its own inference script because they require different state dimensions and state construction logic:

| Level | File | State Dim | Key Features |
|-------|------|-----------|--------------|
| 1 | `infer_rl_a2c_lvl1.py` | 518 | History awareness (visit count, last action, depth) |
| 2 | `infer_rl_a2c_lvl2.py` | 520 | + Redundancy avoidance (spatial overlap detection) |
| 3 | `infer_rl_a2c_lvl3.py` | 1033 | + Contextual memory (parent embedding, hierarchical context) |
| 4 | `infer_rl_a2c_lvl4.py` | 1033 | + Multi-step returns (trained with GAE, same architecture as Level 3) |

## Usage

### Level 1: History Awareness

```bash
python src/inference/a2c/infer_rl_a2c_lvl1.py \
    --image data/to_test_image/test_img_1.svs \
    --model data/models/rl/a2c_lvl1/a2c_lvl1_final.pt \
    --output-viz-path data/visualizations/rl/a2c/viz_a2c_lvl1.html \
    --max-depth 6
```

### Level 2: Redundancy Avoidance

```bash
python src/inference/a2c/infer_rl_a2c_lvl2.py \
    --image data/to_test_image/test_img_1.svs \
    --model data/models/rl/a2c_lvl2/a2c_lvl2_final.pt \
    --output-viz-path data/visualizations/rl/a2c/viz_a2c_lvl2.html \
    --max-depth 6
```

### Level 3: Contextual Memory

```bash
python src/inference/a2c/infer_rl_a2c_lvl3.py \
    --image data/to_test_image/test_img_1.svs \
    --model data/models/rl/a2c_lvl3/a2c_lvl3_final.pt \
    --output-viz-path data/visualizations/rl/a2c/viz_a2c_lvl3.html \
    --max-depth 6
```

### Level 4: Multi-Step Returns with GAE

```bash
python src/inference/a2c/infer_rl_a2c_lvl4.py \
    --image data/to_test_image/test_img_1.svs \
    --model data/models/rl/a2c_lvl4/a2c_lvl4_final.pt \
    --output-viz-path data/visualizations/rl/a2c/viz_a2c_lvl4.html \
    --max-depth 6
```

## Command-Line Arguments

All scripts support the same arguments:

- `--image`: Path to WSI image file (`.svs` format)
- `--model`: Path to trained model checkpoint (`.pt` file)
- `--output-viz-path`: Output path for HTML visualization
- `--max-depth`: Maximum recursion depth for zoom exploration (default: 6)
- `--stochastic`: Use stochastic policy (sample actions). By default, inference is deterministic (greedy)

## Output

Each inference run produces:

1. **Console logs**: Step-by-step decisions (STOP/ZOOM), probabilities, values
2. **HTML visualization**: Interactive visualization of kept/discarded patches (opens in browser automatically)
3. **Metrics**: 
   - Total min-level patches
   - Kept patches count
   - Kept/Min-level ratio (compression metric)

## State Construction Details

### Level 1 (518-D)
```
[env_state(515), visit_count(1), last_action(1), depth(1)]
```

### Level 2 (520-D)
```
[env_state(515), visit_count(1), last_action(1), depth(1), redundancy_score(1), overlap_penalty(1)]
```

### Level 3 & 4 (1033-D)
```
[env_state(515), history(5), parent_embedding(512), has_parent(1)]
```

## Implementation Notes

### History Tracking

All levels maintain a `HistoryTracker` instance that:
- Records visited patches (spatial hashing with grid_size=32)
- Tracks last action and exploration depth
- **Level 2+**: Computes redundancy scores and overlap penalties
- **Level 3+**: Maintains parent-child relationships via embedding storage

### Deterministic vs Stochastic

- **Deterministic** (default): `action = logits.argmax()` - Always chooses highest-probability action
- **Stochastic** (`--stochastic`): `action ~ Categorical(logits)` - Samples from policy distribution

For evaluation and deployment, use deterministic inference for reproducible results.

### Parent Embedding Tracking (Level 3+)

When the agent performs a ZOOM action:
1. Current patch embedding (512-D) is stored as `parent_embedding`
2. `has_parent` flag is set to True
3. Child patches receive this parent context in their state
4. Enables hierarchical decision-making

## Comparison Workflow

To compare all 4 levels on the same image:

```bash
# Run all levels
for lvl in 1 2 3 4; do
    python src/inference/a2c/infer_rl_a2c_lvl${lvl}.py \
        --image data/to_test_image/test_img_1.svs \
        --model data/models/rl/a2c_lvl${lvl}/a2c_lvl${lvl}_final.pt \
        --output-viz-path data/visualizations/rl/a2c/viz_a2c_lvl${lvl}.html \
        --max-depth 6
done
```

Then compare:
- Kept patch counts
- Compression ratios
- Spatial exploration patterns (via HTML visualizations)
- Decision probabilities (console logs)

## Expected Behavior Differences

- **Level 1**: Basic history awareness, may still revisit regions
- **Level 2**: Actively avoids redundant exploration (lower kept count expected)
- **Level 3**: Considers parent-child relationships, more structured zoom tree
- **Level 4**: Similar behavior to Level 3 (training difference, not inference architecture)

## Troubleshooting

### Model loading errors
- Ensure model checkpoint exists at specified path
- Check that model was trained for the correct level (state_dim must match)

### State dimension mismatch
```
RuntimeError: size mismatch for encoder.0.weight
```
- You're using the wrong inference script for your model
- Match script level to model level (e.g., Level 3 model → `infer_rl_a2c_lvl3.py`)

### Memory issues (Level 3+)
- Large state dimension (1033-D) requires more memory
- Reduce `--max-depth` if running out of memory
- Use smaller test images for initial experiments

---

**Created**: February 1, 2026  
**Author**: Jay  
**Repository**: PLIP-dynamic-patcher-MA
