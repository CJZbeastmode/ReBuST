"""Unified supervised preprocessing pipeline for score/zoom regressors."""

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
from tqdm import tqdm

repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.utils.wsi import WSI
from src.utils.embedder import Embedder
from src.utils.dynamic_patch_env import DynamicPatchEnv
from src.inference.supervised.supervised_score_regressor_infer import (
    greedy_infer_zoom_regressor as score_greedy_infer,
    load_regressor as load_score_regressor,
)
from src.inference.supervised.supervised_zoom_classifier_infer import (
    greedy_infer_zoom_regressor as zoom_greedy_infer,
    load_regressor as load_zoom_regressor,
)


DEFAULT_IMAGES_DIR = "/Volumes/Xbox_HD/Data/med_img/train"
DEFAULT_OUTPUT_DIR = "/Volumes/Xbox_HD/Data/extracted/supervised/train"
DEFAULT_SCORE_MODEL = "data/models/supervised/score_regressor.pth"
DEFAULT_ZOOM_MODEL = "data/models/supervised/zoom_classifier.pth"


def parse_label_from_stem(stem: str) -> str:
    return stem.rsplit("-", 1)[-1]


def discover_cases(images_dir: str) -> Dict[str, str]:
    cases: Dict[str, str] = {}
    for fname in sorted(os.listdir(images_dir)):
        if fname.startswith(".") or fname.startswith("._"):
            continue
        if not fname.lower().endswith(".svs"):
            continue
        stem = os.path.splitext(fname)[0]
        cases[stem] = parse_label_from_stem(stem)
    return cases


def _pick_backend(
    supervised_model_type: str,
) -> Tuple[Callable, Callable, str]:
    if supervised_model_type == "score_regressor":
        return load_score_regressor, score_greedy_infer, "supervised_score"
    if supervised_model_type == "zoom_regressor":
        return load_zoom_regressor, zoom_greedy_infer, "supervised_zoom"
    raise ValueError(
        "supervised_model_type must be one of {'score_regressor', 'zoom_regressor'}"
    )


def _safe_state_dim(env: DynamicPatchEnv, level: int, x: int, y: int) -> int:
    try:
        sample_patch = env.wsi.get_patch(level, x, y)
        return len(env.encode_state(sample_patch, lvl=level, x=x, y=y))
    except Exception:
        return 515


@torch.no_grad()
def _infer_root_only(
    env: DynamicPatchEnv,
    model: torch.nn.Module,
    level: int,
    x: int,
    y: int,
    device: torch.device,
) -> Tuple[List[Tuple[object, Dict]], List[Tuple[object, Dict]]]:
    kept, discarded = [], []

    patch = env.wsi.get_patch(level, x, y)
    state = env.encode_state(patch, lvl=level, x=x, y=y)
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    score_stop, score_zoom = model(state_tensor).squeeze(0).tolist()
    decision = env.infer_zoom_decision(score_stop, score_zoom)

    meta = {
        "level": int(level),
        "x": int(x),
        "y": int(y),
        "score": float(score_stop if decision == 0 else score_zoom),
    }
    if decision == 0:
        kept.append((patch, meta))
    else:
        discarded.append((patch, meta))
    return kept, discarded


def _to_patch_dict(
    items: List[Tuple[object, Dict]]
) -> Dict[Tuple[int, int, int], Dict]:
    out: Dict[Tuple[int, int, int], Dict] = {}
    for _patch, meta in items:
        key = (int(meta["level"]), int(meta["x"]), int(meta["y"]))
        out[key] = {"score": float(meta.get("score", 0.0))}
    return out


def _infer_case(args, case_id: str, img_path: str):
    load_model_fn, greedy_fn, method_prefix = _pick_backend(args.supervised_model_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = Embedder(img_backend="plip")
    wsi = WSI(img_path, embedder=embedder)
    env = DynamicPatchEnv(wsi)

    root_level = int(env.max_level)
    width, height = wsi.levels_info[root_level]["size"]

    state_dim = _safe_state_dim(env, root_level, 0, 0)
    model = load_model_fn(args.model, device, state_dim=state_dim)

    kept_all: List[Tuple[object, Dict]] = []
    disc_all: List[Tuple[object, Dict]] = []

    for y in range(0, height, wsi.patch_size):
        for x in range(0, width, wsi.patch_size):
            try:
                if args.level_mode == "root_only":
                    kept, disc = _infer_root_only(env, model, root_level, x, y, device)
                else:
                    kept, disc = greedy_fn(
                        env,
                        model,
                        root_level,
                        x,
                        y,
                        max_depth=args.max_depth,
                        device=device,
                    )
                kept_all.extend(kept)
                disc_all.extend(disc)
            except Exception:
                continue

    active_patches = _to_patch_dict(kept_all)
    zoomed_patches = _to_patch_dict(disc_all)

    return {
        "case_id": case_id,
        "active_patches": active_patches,
        "zoomed_patches": zoomed_patches,
        "levels_info": wsi.levels_info,
        "patch_size": wsi.patch_size,
        "multistage": wsi.multistage,
        "patch_count": len(active_patches),
        "zoomed_count": len(zoomed_patches),
        "method": f"{method_prefix}_{args.level_mode}",
    }


def main(args) -> None:
    images_dir = os.path.abspath(args.images_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Supervised model checkpoint not found: {args.model}")

    cases = discover_cases(images_dir)
    if not cases:
        print(f"[WARN] No .svs files found in {images_dir}")
        return

    success = skipped = failed = 0
    for case_id in tqdm(sorted(cases.keys()), desc="supervised_preprocess"):
        out_path = os.path.join(output_dir, f"{case_id}.pt")
        if os.path.exists(out_path) and not args.force:
            skipped += 1
            continue

        img_path = os.path.join(images_dir, f"{case_id}.svs")
        if not os.path.exists(img_path):
            failed += 1
            continue

        try:
            result = _infer_case(args, case_id, img_path)
            torch.save(
                {
                    "case_id": case_id,
                    "label": cases[case_id],
                    "img_path": img_path,
                    "active_patches": result["active_patches"],
                    "zoomed_patches": result["zoomed_patches"],
                    "levels_info": result["levels_info"],
                    "patch_size": result["patch_size"],
                    "multistage": result["multistage"],
                    "patch_count": result["patch_count"],
                    "zoomed_count": result["zoomed_count"],
                    "method": result["method"],
                },
                out_path,
            )
            if args.verbose:
                print(
                    f"[SAVED] {case_id}: patch_count={result['patch_count']} "
                    f"zoomed_count={result['zoomed_count']}"
                )
            success += 1
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {case_id}: {exc}")
            if args.verbose:
                traceback.print_exc()

    print(f"[DONE] success={success} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified supervised preprocessing pipeline"
    )
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_SCORE_MODEL)
    parser.add_argument(
        "--supervised-model-type",
        choices=["score_regressor", "zoom_regressor"],
        default="score_regressor",
    )
    parser.add_argument(
        "--level-mode",
        choices=["root_only", "finest_only"],
        default="finest_only",
    )
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if (
        args.model == DEFAULT_SCORE_MODEL
        and args.supervised_model_type == "zoom_regressor"
    ):
        args.model = DEFAULT_ZOOM_MODEL

    main(args)
