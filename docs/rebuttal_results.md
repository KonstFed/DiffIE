# DiffIE — Rebuttal Experiments (new results)

Main model: `configs/lsoie_ex_2500` (DiffIE, D3PM-uniform, T=16).
Scorer: `benchmarks/CaRB/carb.py`; **CaRB** = default matcher, **CaRB(1-1)** = `--binary`.
All F1/AUC are CaRB optimal-point values.

---

## 1. Non-diffusion control: MC-dropout tagger (attribution)

Same encoder (`bert-base-uncased`), same B/S/R/O labels, same longest-span
decode, same lenient-frequency clustering. Candidates drawn by full-encoder
MC-dropout instead of reverse diffusion. Inference operating point tuned on CaRB
dev over a full grid — n ∈ {1..512}, k ∈ {1,2,3,4,6,8,10}, τ ∈ {0.7,0.8,0.9}
(280 configs); best configuration reported.

| System (CaRB dev)        | F1    | P     | R     | AUC   | config          |
|--------------------------|------:|------:|------:|------:|-----------------|
| MC-dropout tagger (best) | 0.375 | 0.689 | 0.258 | 0.213 | n=256,k=10,τ=0.7 |
| DiffIE                   | 0.523 | 0.610 | 0.458 | 0.370 | n=512,k=4,τ=0.9  |

| System (CaRB 1-1 dev)    | F1    | P     | R     | AUC   | config          |
|--------------------------|------:|------:|------:|------:|-----------------|
| MC-dropout tagger (best) | 0.373 | 0.664 | 0.259 | 0.200 | n=256,k=10,τ=0.7 |
| DiffIE                   | 0.531 | 0.593 | 0.480 | 0.361 | n=512,k=4,τ=0.9  |

- Diffusion-attributable margin: **+0.148 F1** (CaRB dev), driven by recall (0.458 vs 0.258).
- Tagger is **pool-limited**: over the entire 280-config grid its F1 stays in
  0.296–0.375 and recall never exceeds **0.273** (CaRB) / **0.279** (CaRB 1-1),
  even at n=512, k=10. Re-clustering (k, τ) cannot recover triplets absent from the pool.
- Matched-n: tagger best F1 = 0.360 (n=16) → 0.375 (n=512); DiffIE rises 0.493 → 0.523 over the same n.

---

## 2. DiffIE quality vs sample count n (CaRB dev, direct rerun)

| n   | CaRB F1 | CaRB AUC | CaRB(1-1) F1 | CaRB(1-1) AUC |
|-----|--------:|---------:|-------------:|--------------:|
| 16  | 0.493   | 0.348    | 0.500        | 0.338         |
| 64  | 0.508   | 0.364    | 0.518        | 0.355         |
| 512 | 0.523   | 0.370    | 0.531        | 0.361         |

Near-peak quality (0.493, ~94% of peak F1) is reached by n=16.

We have same in the figure 2 in paper

---

## 3. Hyperparameter transfer: τ/k/n on BenchIE & WiRe57

Operating point selected on CaRB dev (n=512, k=4, τ=0.9), then applied without
retuning; each of n, k, τ swept on the other benchmarks (test).

| Benchmark | DiffIE F1 @ transferred pt | best F1 in sweep | gap   |
|-----------|---------------------------:|-----------------:|------:|
| CaRB dev  | 0.523                      | 0.523 (selected) | —     |
| BenchIE   | 0.343                      | ~0.363 (τ=0.95)  | 0.02  |
| WiRe57    | 0.361                      | ~0.361 (k=4/6)   | ~0.00 |

The CaRB-dev point is at/near the per-benchmark optimum → gains are not a
CaRB-matcher alignment artifact.

### Full sensitivity sweep (F1; `*` = transferred operating point n=512, k=4, τ=0.9)

**Threshold τ** (n=512, k=4)

| τ    | CaRB dev  | BenchIE   | WiRe57    |
|------|----------:|----------:|----------:|
| 0.50 | 0.439     | 0.222     | 0.186     |
| 0.60 | 0.448     | 0.229     | 0.196     |
| 0.70 | 0.473     | 0.250     | 0.257     |
| 0.80 | 0.517     | 0.288     | 0.305     |
| 0.90\* | **0.523** | 0.343     | **0.361** |
| 0.95 | 0.498     | **0.363** | 0.348     |

**Top-k extractions** (n=512, τ=0.9)

| k    | CaRB dev  | BenchIE   | WiRe57    |
|------|----------:|----------:|----------:|
| 1    | 0.434     | 0.200     | 0.174     |
| 2    | 0.518     | 0.302     | 0.270     |
| 4\*  | **0.523** | **0.343** | **0.361** |
| 6    | 0.522     | 0.319     | 0.361     |
| 8    | 0.522     | 0.294     | 0.344     |
| 10   | 0.522     | 0.271     | 0.324     |

**Samples per sentence n** (k=4, τ=0.9)

| n    | CaRB dev  | BenchIE   | WiRe57    |
|------|----------:|----------:|----------:|
| 1    | 0.396     | 0.137     | 0.153     |
| 2    | 0.454     | 0.199     | 0.197     |
| 4    | 0.474     | 0.258     | 0.292     |
| 8    | 0.475     | 0.282     | 0.302     |
| 16   | 0.493     | 0.305     | 0.328     |
| 32   | 0.500     | 0.319     | 0.332     |
| 64   | 0.508     | 0.339     | 0.343     |
| 128  | 0.513     | 0.347     | 0.357     |
| 256  | 0.517     | 0.351     | 0.360     |
| 512\* | **0.523** | 0.343     | **0.361** |

- Flat for **k≥4** (CaRB 0.522–0.523) and **n≥64** (CaRB 0.508–0.523): insensitive to these choices.
- τ is the main knob, monotonic to 0.9; default τ=0.9 is near-optimal without retuning.
- Off-domain peaks (BenchIE τ=0.95/n=256) barely beat the transferred point → not over-fit to eval benchmarks.

Plots: `artifacts/sweep_all_params_on_test/sensitivity_curves.pdf` (and `.png`).

---

## 4. Efficiency: throughput / latency / memory (A100)

Inference-only, averaged over 3+ repeats. DetIE and Qwen on all 641 CaRB-dev
sentences; DiffIE and tagger on a 200-sentence timing subset. VRAM for
DiffIE/tagger via `torch.cuda.max_memory_allocated`.

| System            |   n | sent/s | F1 (CaRB dev) | Peak VRAM        |
|-------------------|----:|-------:|--------------:|------------------|
| DetIE-LSOIE       |   – | 685.96 | 0.451         | ~0.7 GB (est.)   |
| MC-dropout tagger |  16 |  48.40 | 0.360         | 0.46 GB          |
| DiffIE            |  16 |  14.23 | 0.493         | 0.55 GB          |
| DiffIE            |  64 |   6.29 | 0.508         | 0.65 GB          |
| MC-dropout tagger | 512 |   1.93 | 0.375         | 0.75 GB          |
| DiffIE            | 512 |   0.95 | 0.523         | 1.74 GB          |
| Qwen3-30B-A3B     |   – |  0.283 | 0.495         | ~80 GB (1×A100)  |

- Clustering/ranking is **0.21%** of DiffIE runtime at n=16, **1.01%** at n=512;
  cost is almost entirely reverse-diffusion sampling.
- DetIE VRAM estimated from model size (bert-base-multilingual-cased, ~178M).
- Qwen's ~80 GB is vLLM's KV-cache reservation, not intrinsic model footprint.
