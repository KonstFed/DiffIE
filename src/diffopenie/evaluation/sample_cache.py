"""Sample N raw triplets per sentence once and dump them to JSONL.

The heavy cost in CaRB eval is the model's reverse-diffusion sampling. Caching
the raw samples lets `diffopenie.plotting.lsoie_cache_curve_plot` sweep `n`
without re-running the model — this is how Figure 1 of the paper is produced.

JSONL row schema:
    {"sentence": str,
     "words": [str, ...],
     "samples": [[[s0,s1] | null, [o0,o1] | null, [p0,p1] | null], ...]}

Usage:
    uv run python -m diffopenie.evaluation.sample_cache \
        --config configs/lsoie_ex_2500/lsoie_ex_2500_config.yaml \
        --checkpoint-path configs/lsoie_ex_2500/weights.pt \
        --input-sentences benchmarks/CaRB/data/dev.txt \
        --n 1024 \
        --out configs/lsoie_ex_2500/n_study/cache_dev_1024.jsonl
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from diffopenie.evaluation.carb_eval import load_model
from diffopenie.training.train_example import TrainingConfig
from diffopenie.utils import load_config


def _span_to_list(s: tuple[int, int] | None) -> list[int] | None:
    return None if s is None else [int(s[0]), int(s[1])]


@torch.no_grad()
def sample_and_dump(
    model,
    sentences: list[str],
    n: int,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with open(out_path, "w", encoding="utf-8") as f:
        for sent in tqdm(sentences, desc="Sampling"):
            words = sent.split()
            triplets = model.get_triplets([words], n=n)
            samples = [
                [_span_to_list(sub), _span_to_list(obj), _span_to_list(pred)]
                for (sub, obj, pred) in triplets
            ]
            f.write(
                json.dumps(
                    {"sentence": sent, "words": words, "samples": samples},
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sample N raw triplets per sentence and cache to JSONL"
    )
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint-path", type=Path, required=True)
    ap.add_argument(
        "--input-sentences",
        type=Path,
        required=True,
        help="One sentence per line (CaRB dev.txt / test.txt)",
    )
    ap.add_argument(
        "--n", type=int, default=1024, help="Samples per sentence (default 1024)"
    )
    ap.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    args = ap.parse_args()

    print("Loading model...")
    config = load_config(TrainingConfig, args.config)
    model = load_model(config, args.checkpoint_path)

    with open(args.input_sentences, "r", encoding="utf-8") as f:
        sentences = [ln.strip() for ln in f if ln.strip()]
    print(f"Sampling {args.n} triplets per sentence ({len(sentences)} sentences)...")

    sample_and_dump(model, sentences, args.n, args.out)
    print(f"Cache written to {args.out}")


if __name__ == "__main__":
    main()
