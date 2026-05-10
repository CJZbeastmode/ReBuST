# MA (reorganized)

Quick notes:

- Code is maintained in `src/` as before. For convenience there's a `ma/` package that exposes many modules from `src/` so you can import them as `from ma.reward_module import InfoGainReward`.
- Utility scripts were copied into `scripts/` (you can run them directly).
- The `dynamic_patcher` utilities were archived into `src/archive/dynamic_patcher/` to keep them safe.

Recommended quick start (use your conda env):

```fish
# from repository root
python -m rl        # original entrypoint (keeps using src/ modules)
# or run scripts explicitly
python scripts/build_faiss_txt.py
```



To mention:
PLIP + plip result

## Validate information-gain proxies

Use this script to compare entropy/contrastive/centroid proxy scores against
actual downstream label gain with MI proxies and rank correlation.

```fish
python scripts/validate_info_gain_proxy.py \
	--csv data/benchmark/proxy_eval.csv \
	--score-cols entropy_score contrastive_score centroid_score \
	--gain-col downstream_gain \
	--out-csv data/benchmark/proxy_eval_ranked.csv \
	--out-json data/benchmark/proxy_eval_ranked.json \
	--plot-dir data/visualizations/proxy_eval
```

If your CSV does not have a direct gain column, pass before/after metrics:

```fish
python scripts/validate_info_gain_proxy.py \
	--csv data/benchmark/proxy_eval.csv \
	--score-cols entropy_score contrastive_score centroid_score \
	--before-col accuracy_before \
	--after-col accuracy_after
```