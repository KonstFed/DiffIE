# DiffIE: Diffusion-based Open Information Extraction

Code accompanying the paper *DiffIE: Diffusion-based Open Information
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

- IMoJIE training data (Zenodo 3775983) → `dataset/data/`
- CaRB → `benchmarks/CaRB/`
- WiRe57 → `benchmarks/WiRe57/`
- BenchIE → `benchmarks/benchie/`
- CycleOIE's curated LSOIE-G subsets → `dataset/cycleoie/`

Python 3.12 is required (see `pyproject.toml`).

## Repository layout

```
configs/        one directory per paper experiment; each contains a self-contained
                YAML and (after training) a weights.pt checkpoint.
scripts/        bash entry points:
                  train_all.sh           train all 9 configs
                  eval_test_n_times.sh   CaRB-test + BenchIE + WiRe57 over 10 seeds
                  eval_dev_n_times.sh    CaRB-dev predictions over 10 seeds, all configs
                  eval_dev.sh            CaRB-dev predictions, single seed, all configs
                  score_test_n_times.sh  print test-set metrics from cached extractions
                  score_dev_n_times.sh   print dev metrics from cached extractions
                  score_dev.sh           run the real CaRB scorer on dev predictions
                  plot_n_study.sh        Figure 1 — n-study cache + curve plot
                  make_tagger_configs.py generate the MC-dropout tagger configs
                  sweep_k.py             re-cluster a cached pool over k/n/tau
                  sweep_benchie_wire57_sensitivity.py
                                         tau/k/n transfer study on all benchmarks
                  plot_quality_cost.py   quality-cost curve (F1 vs seconds/sentence)
baselines/      DetIE setup and timing runners (see baselines/README.md).
src/diffopenie/ Python package: data, diffusion kernels, encoder + denoiser, training,
                evaluation, plotting.
tests/          unit tests for span handling, CaRB metrics, label mapping.
```

Every config under `configs/` follows the same convention: the directory name
matches the paper subset (`lsoie_ex_50`, `lsoie_ex_full`, etc.) and contains
`<name>_config.yaml`. After training, `weights.pt` is dropped next to the YAML.

## Artifacts

Trained weights, cached per-seed predictions, and the n-study sample caches are
hosted on Hugging Face and mirror the `configs/` layout:

```bash
./scripts/fetch_artifacts.sh
```

Set `REPO` in that script to the artifact repository. It is private during
review; log in once with a read token:

```bash
uvx --from "huggingface_hub[cli]" hf auth login
```

## Streamlit demo

```bash
uv run streamlit run src/diffopenie/evaluation/app.py
```

## Reproducing the paper

The primary model (Table 1, also used for Figure 1 and the appendix) is
`configs/lsoie_ex_2500/`. All `scripts/*` commands expect to be run from the
repository root and assume the data setup above is complete.

### Train every checkpoint (Tables 1, 2, 3)

```bash
./scripts/train_all.sh
```

This sequentially runs each config (data ablation + MDLM ablation + primary)
and moves the best CaRB-dev checkpoint to `configs/<name>/weights.pt`. To run
a subset:

```bash
./scripts/train_all.sh --only 2500,2500_mdlm
```

Or one config at a time:

```bash
uv run python -m diffopenie.training.train_example configs/lsoie_ex_2500/lsoie_ex_2500_config.yaml
```

### Table 1 — main results across four benchmarks (10 seeds)

```bash
./scripts/eval_test_n_times.sh configs/lsoie_ex_2500
./scripts/score_test_n_times.sh configs/lsoie_ex_2500
```

The first command produces per-seed extractions for CaRB / CaRB (1-1) /
BenchIE / WiRe57; the second prints mean ± std over the 10 seeds.

### Table 2 — data-ablation, CaRB-dev (10 seeds × 7 configs)

```bash
./scripts/eval_dev_n_times.sh
./scripts/score_dev_n_times.sh
```

The first command iterates over every `configs/*/` that has a `weights.pt`;
the second prints per-config mean ± std for CaRB and CaRB (1-1).

### Table 3 — MDLM vs. D3PM-uniform ablation

Train both MDLM configs (covered by `train_all.sh`):

```bash
./scripts/train_all.sh --only 2500_mdlm,full_mdlm
```

Then read the MDLM rows from the same dev-score table:

```bash
./scripts/score_dev_n_times.sh
```

### Figure 1 — test-time compute scaling

```bash
./scripts/plot_n_study.sh configs/lsoie_ex_2500 1024
```

The first run caches 1024 raw triplets per CaRB-dev sentence to
`configs/lsoie_ex_2500/n_study/cache_dev_1024.jsonl` (slow). Subsequent runs
reuse the cache and only re-render the plot. Output:
`configs/lsoie_ex_2500/n_study/curve{.csv,.png,.pdf}` plus the F1-only variant.

### MC-dropout tagger control (non-diffusion baseline)

Same encoder, label space, span decoding and clustering as DiffIE; reverse
diffusion is replaced by full-encoder MC dropout, so the candidate generator is
the only difference. Configs are generated from the diffusion ones:

```bash
uv run python scripts/make_tagger_configs.py
uv run python -m diffopenie.training.train_example \
    configs/lsoie_ex_tagger_2500/lsoie_ex_tagger_2500_config.yaml
```

Evaluation reuses `carb_eval` unchanged. To sweep the operating point over a
cached candidate pool without re-running the model:

```bash
uv run python scripts/sweep_k.py \
    --config configs/lsoie_ex_tagger_2500/lsoie_ex_tagger_2500_config.yaml \
    --cache configs/lsoie_ex_tagger_2500/n_study/cache_dev_512.jsonl \
    --out configs/lsoie_ex_tagger_2500/n_study/k_sweep.json \
    --ks 1 2 3 4 6 8 10
```

### tau / k / n sensitivity across benchmarks

Holds the CaRB-dev-selected operating point (n=512, k=4, tau=0.9) fixed and
varies one parameter at a time on CaRB dev, BenchIE and WiRe57:

```bash
uv run python scripts/sweep_benchie_wire57_sensitivity.py \
    --config configs/lsoie_ex_2500/lsoie_ex_2500_config.yaml \
    --checkpoint-path configs/lsoie_ex_2500/weights.pt \
    --out-dir configs/lsoie_ex_2500/sensitivity \
    --max-n 512
```

### Efficiency — throughput, latency, memory

```bash
uv run python -m diffopenie.evaluation.timing_benchmark \
    --config configs/lsoie_ex_2500/lsoie_ex_2500_config.yaml \
    --checkpoint-path configs/lsoie_ex_2500/weights.pt \
    --out timing_diffie.csv
```

`scripts/timing_a100.slurm` runs the same sweep on a single A100. Baselines:
DetIE via `baselines/` (setup, evaluation and timing scripts), and a prompted
LLM via `scripts/serve_qwen_vllm.slurm` plus
`diffopenie.evaluation.llm_extract`. `scripts/plot_quality_cost.py` renders F1
against measured cost per sentence.

## Free-form prediction

```bash
uv run python -m diffopenie.evaluation.generate_cli \
    --config configs/lsoie_ex_2500/lsoie_ex_2500_config.yaml \
    --checkpoint-path configs/lsoie_ex_2500/weights.pt \
    --sentences "Marie Curie discovered radium in 1898 ."
```

## Hyperparameters

Every experiment is fully specified by its `configs/<name>/<name>_config.yaml`.
The primary configuration is `configs/lsoie_ex_2500/`; MDLM variants live in
`configs/lsoie_ex_*_mdlm/`.
