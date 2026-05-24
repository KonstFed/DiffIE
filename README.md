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
src/diffopenie/ Python package: data, diffusion kernels, encoder + denoiser, training,
                evaluation, plotting.
tests/          unit tests for span handling, CaRB metrics, label mapping.
```

Every config under `configs/` follows the same convention: the directory name
matches the paper subset (`lsoie_ex_50`, `lsoie_ex_full`, etc.) and contains
`<name>_config.yaml`. After training, `weights.pt` is dropped next to the YAML.

## Artifacts

Trained weights and cached predictions are not provided yet but will be soon

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
