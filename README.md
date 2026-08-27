# DiffIE: Diffusion-based Open Information Extraction

Code and checkpoints for the paper *DiffIE: Diffusion-based Open Information
Extraction*. DiffIE formulates Open Information Extraction as conditional
discrete diffusion over per-token role tags. At inference, independent
reverse-diffusion trajectories produce a candidate pool that a lenient-match
extractor clusters into the final triplet set.

## Setup

```bash
uv sync
./load_data.sh
```

`load_data.sh` fetches:

- CaRB → `benchmarks/CaRB/`
- WiRe57 → `benchmarks/WiRe57/`
- BenchIE → `benchmarks/benchie/`
- CycleOIE's curated LSOIE-G subsets → `dataset/cycleoie/`

The `plus20k` / `plus_full` configs additionally mix in original LSOIE, which
`datasets` pulls from the Hub (`wardenga/lsoie`) on first use — no manual step.

Python 3.12 is required (see `pyproject.toml`).

## Repository layout

```
configs/        one directory per paper experiment; each contains a self-contained
                YAML and (after training or fetch_artifacts.sh) a weights.pt.
scripts/        bash entry points:
                  train_all.sh           train all 16 configs (9 diffusion + 7 tagger)
                  fetch_artifacts.sh     pull released weights/predictions/caches
                  eval_test_n_times.sh   CaRB-test + BenchIE + WiRe57 over 10 seeds
                  eval_dev_n_times.sh    CaRB-dev predictions over 10 seeds, all configs
                  eval_dev.sh            CaRB-dev predictions, single seed, all configs
                  score_test_n_times.sh  print test-set metrics from cached extractions
                  score_dev_n_times.sh   print dev metrics from cached extractions
                  score_dev.sh           run the real CaRB scorer on dev predictions
                  plot_n_study.sh        sample-count scaling cache + curve (Table 10)
                  make_tagger_configs.py generate the MC-dropout tagger configs
                  sweep_k.py             re-cluster a cached pool over k/n/tau
                  sweep_benchie_wire57_sensitivity.py
                                         tau/k/n transfer study on all benchmarks
                  plot_quality_cost.py   F1 vs measured seconds/sentence
                  timing_a100.slurm      efficiency sweep on a single A100
                  serve_qwen_vllm.slurm  vLLM server for the prompted-LLM baseline
baselines/      DetIE setup, evaluation and timing runners (see baselines/README.md).
docs/           rebuttal_results.md — supplementary numbers behind Tables 4 and 6.
src/diffopenie/ Python package: data, diffusion kernels, encoder + denoiser, training,
                evaluation, plotting.
tests/          unit tests for span handling, CaRB metrics, label mapping.
```

Every config under `configs/` follows the same convention: the directory name
matches the paper subset (`lsoie_ex_50`, `lsoie_ex_full`, etc.) and contains
`<name>_config.yaml`. After training, `weights.pt` is dropped next to the YAML.
`lsoie_ex_tagger_*` are the matched non-diffusion controls (Table 6).

## Artifacts

Trained weights, cached per-seed predictions, and the sample-count caches are
hosted on Hugging Face and mirror the `configs/` layout:

```bash
./scripts/fetch_artifacts.sh
```

The repository is [KonstFed/diffIE](https://huggingface.co/KonstFed/diffIE);
override it with `DIFFIE_HF_REPO` if you mirror it elsewhere.

`weights.pt` is released for `lsoie_ex_2500` (the primary model),
`lsoie_ex_full` and `lsoie_ex_full_mdlm`, along with the 1024-sample caches for
`lsoie_ex_2500` and `lsoie_ex_full`. The remaining data-ablation variants ship
cached per-seed extractions only, so `score_dev_n_times.sh` reproduces Table 3
in full while `eval_dev_n_times.sh` will skip them; retrain those with
`./scripts/train_all.sh --only <name>` if you need the checkpoints. Tagger
baselines are trained from their configs (see below) — no checkpoint is shipped.

## Interactive demo

Type a sentence, watch the reverse-diffusion trajectory denoise it into role
tags step by step, and read off the clustered triplets.

```bash
./scripts/fetch_artifacts.sh    # once — pulls the checkpoint (~1.1 GB)
uv run streamlit run src/diffopenie/evaluation/app.py
```

It defaults to the primary model (`configs/lsoie_ex_2500`). To point it at
another checkpoint, pass arguments after Streamlit's `--` separator:

```bash
uv run streamlit run src/diffopenie/evaluation/app.py -- \
    --config configs/lsoie_ex_full_mdlm/lsoie_ex_full_mdlm_config.yaml \
    --checkpoint configs/lsoie_ex_full_mdlm/weights.pt
```

The sidebar exposes the inference-time knobs from the paper — sample count `n`,
returned triplets `k`, and the lenient-match threshold `tau` — so you can watch
quality and cost trade off live. Streamlit binds every interface by default; add
`--server.address localhost` to keep it local.

## Reproducing the paper

The primary model is `configs/lsoie_ex_2500` (`lsoie-ex-2.5k` in the paper),
evaluated at the CaRB-dev-selected operating point **n=512, k=4, tau=0.9**,
which is held fixed for every reported test number. All `scripts/*` commands
expect to be run from the repository root and assume the data setup above is
complete.

### Train every checkpoint

```bash
./scripts/train_all.sh
```

This sequentially runs each config — the seven data-ablation configs, the two
MDLM ablations, and the seven MC-dropout tagger controls — and moves the best
CaRB-dev checkpoint to `configs/<name>/weights.pt`. To run a subset:

```bash
./scripts/train_all.sh --only 2500,2500_mdlm
```

Or one config at a time:

```bash
uv run python -m diffopenie.training.train_example configs/lsoie_ex_2500/lsoie_ex_2500_config.yaml
```

Tagger configs must be generated before they can be trained:

```bash
uv run python scripts/make_tagger_configs.py
```

### Table 2 — main results across four benchmarks (10 seeds)

```bash
./scripts/eval_test_n_times.sh configs/lsoie_ex_2500
./scripts/score_test_n_times.sh configs/lsoie_ex_2500
```

The first command produces per-seed extractions for CaRB / CaRB (1-1) /
BenchIE / WiRe57; the second prints mean ± std over the 10 seeds. Per-seed
values are reported in Appendix Tables 7 and 8.

### Table 3 — data-ablation, CaRB-dev (10 seeds × 7 configs)

```bash
./scripts/eval_dev_n_times.sh
./scripts/score_dev_n_times.sh
```

The first command iterates over every `configs/*/` that has a `weights.pt`;
the second prints per-config mean ± std for CaRB and CaRB (1-1).

### Table 5 — D3PM-uniform vs. MDLM (absorbing state)

Train both MDLM configs (covered by `train_all.sh`):

```bash
./scripts/train_all.sh --only 2500_mdlm,full_mdlm
```

Then read the MDLM rows from the same dev-score table:

```bash
./scripts/score_dev_n_times.sh
```

### Table 4 — efficiency: throughput, latency, memory

```bash
uv run python -m diffopenie.evaluation.timing_benchmark \
    --config configs/lsoie_ex_2500/lsoie_ex_2500_config.yaml \
    --checkpoint-path configs/lsoie_ex_2500/weights.pt \
    --input-sentences benchmarks/CaRB/data/dev.txt \
    --out timing_diffie.csv
```

`scripts/timing_a100.slurm` runs the same sweep on the single A100 used for the
reported numbers. The two reference systems in the table:

- **DetIE-LSOIE** — `baselines/` holds setup, evaluation and timing scripts.
  It is scored under our evaluator, so it shares a scorer but not annotation
  targets with DiffIE (see the paper's inference-cost paragraph).
- **Qwen3-30B-A3B (prompted)** — `scripts/serve_qwen_vllm.slurm` plus
  `diffopenie.evaluation.llm_extract`.

`scripts/plot_quality_cost.py` renders F1 against measured cost per sentence
for any pair of timing CSV + F1 source.

### Table 6 — MC-dropout tagger control (is diffusion necessary?)

Same encoder, label space, span decoding and clustering as DiffIE; reverse
diffusion is replaced by full-encoder MC dropout, so the candidate generator is
the only difference. Configs are generated from the diffusion ones:

```bash
uv run python scripts/make_tagger_configs.py
uv run python -m diffopenie.training.train_example \
    configs/lsoie_ex_tagger_2500/lsoie_ex_tagger_2500_config.yaml
```

Evaluation reuses `carb_eval` unchanged. The tagger's operating point is tuned
over its own n/k/tau grid rather than inheriting DiffIE's; to sweep it over a
cached candidate pool without re-running the model:

```bash
uv run python scripts/sweep_k.py \
    --config configs/lsoie_ex_tagger_2500/lsoie_ex_tagger_2500_config.yaml \
    --cache configs/lsoie_ex_tagger_2500/n_study/cache_dev_512.jsonl \
    --out configs/lsoie_ex_tagger_2500/n_study/k_sweep.json \
    --ks 1 2 3 4 6 8 10
```

Full grid results are in `docs/rebuttal_results.md`.

### Table 9 — tau / k sensitivity across benchmarks

Holds the CaRB-dev-selected operating point (n=512, k=4, tau=0.9) fixed and
varies one parameter at a time on CaRB dev, BenchIE and WiRe57. BenchIE and
WiRe57 have no development split, so these sweeps are diagnostic only and do
not select any reported setting.

```bash
uv run python scripts/sweep_benchie_wire57_sensitivity.py \
    --config configs/lsoie_ex_2500/lsoie_ex_2500_config.yaml \
    --checkpoint-path configs/lsoie_ex_2500/weights.pt \
    --out-dir configs/lsoie_ex_2500/sensitivity \
    --max-n 512
```

### Table 10 — test-time compute scaling (sample count n)

```bash
./scripts/plot_n_study.sh configs/lsoie_ex_2500 1024
```

The first run caches 1024 raw triplets per CaRB-dev sentence to
`configs/lsoie_ex_2500/n_study/cache_dev_1024.jsonl` (slow; `fetch_artifacts.sh`
ships this cache). Subsequent runs reuse the cache and only re-sweep n. Output:
`configs/lsoie_ex_2500/n_study/curve{.csv,.png,.pdf}` — the CSV carries the
lenient-match and exact-frequency columns of Table 10. The BenchIE and WiRe57
columns of that table come from the sensitivity sweep above.

## Free-form prediction

```bash
uv run python -m diffopenie.evaluation.generate_cli \
    --config configs/lsoie_ex_2500/lsoie_ex_2500_config.yaml \
    --checkpoint configs/lsoie_ex_2500/weights.pt \
    --text "Marie Curie discovered radium in 1898 ."
```

## Hyperparameters

Every experiment is fully specified by its `configs/<name>/<name>_config.yaml`,
matching the implementation details in the paper's Experimental Setup:
`bert-base-uncased` encoder with the four lowest layers frozen, a 6-layer
denoiser (d=512, 8 heads, concat fusion), T=16 diffusion steps with a cosine
schedule (s=0.002), AdamW at 2e-4 (denoiser) / 5e-5 (encoder), weight decay
0.01, 500 warmup steps, batch size 32, 12 epochs, background-label class weight
0.7.

The primary configuration is `configs/lsoie_ex_2500/`; MDLM variants live in
`configs/lsoie_ex_*_mdlm/` and the non-diffusion controls in
`configs/lsoie_ex_tagger_*/`.
