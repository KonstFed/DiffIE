# One-sample one-sided t-test comparing DiffIE (10 seeds, 38-47) against published
# single-point baseline values on three metrics where we claim superiority.
# Because baselines report only a single number (no per-seed distribution), we treat
# them as fixed constants and test H0: mu_DiffIE <= baseline vs H1: mu_DiffIE > baseline.
# p < 0.05 means the observed DiffIE mean is unlikely under H0 given our seed variance.

import numpy as np
from scipy.stats import ttest_1samp

# fmt: off
per_seed = {
    "CaRB F1": {
        "auc": np.array([0.3740, 0.3690, 0.3720, 0.3710, 0.3720, 0.3700, 0.3690, 0.3700, 0.3720, 0.3730]),
        "precision": np.array([0.6320, 0.6310, 0.6570, 0.6620, 0.6250, 0.6450, 0.6320, 0.6490, 0.6260, 0.6510]),
        "recall":    np.array([0.4480, 0.4450, 0.4310, 0.4340, 0.4470, 0.4400, 0.4430, 0.4370, 0.4480, 0.4380]),
        "f1":        np.array([0.5240, 0.5220, 0.5210, 0.5240, 0.5210, 0.5230, 0.5210, 0.5220, 0.5220, 0.5240]),
    },
    "CaRB (1-1) F1": {
        "auc":       np.array([0.3460, 0.3430, 0.3440, 0.3470, 0.3450, 0.3440, 0.3440, 0.3440, 0.3420, 0.3460]),
        "precision": np.array([0.5980, 0.6210, 0.5890, 0.6090, 0.5980, 0.5990, 0.6080, 0.6030, 0.6030, 0.5940]),
        "recall":    np.array([0.4610, 0.4430, 0.4610, 0.4600, 0.4580, 0.4570, 0.4530, 0.4540, 0.4510, 0.4590]),
        "f1":        np.array([0.5210, 0.5170, 0.5170, 0.5240, 0.5180, 0.5180, 0.5190, 0.5180, 0.5160, 0.5180]),
    },
    "BenchIE": {
        "precision": np.array([0.3856, 0.3853, 0.3786, 0.3897, 0.3828, 0.3821, 0.3856, 0.3750, 0.3847, 0.3893]),
        "recall":    np.array([0.3096, 0.3074, 0.3096, 0.3126, 0.3096, 0.3096, 0.3133, 0.3044, 0.3089, 0.3126]),
        "f1":        np.array([0.3435, 0.3420, 0.3407, 0.3469, 0.3423, 0.3421, 0.3457, 0.3361, 0.3426, 0.3468]),
    },
    "WiRe57": {
        "precision": np.array([0.4220, 0.4212, 0.4407, 0.4311, 0.4286, 0.4394, 0.4268, 0.4346, 0.4300, 0.4273]),
        "recall":    np.array([0.3043, 0.3071, 0.3131, 0.3078, 0.3048, 0.3195, 0.3152, 0.3097, 0.3105, 0.3166]),
        "f1":        np.array([0.3536, 0.3552, 0.3661, 0.3592, 0.3563, 0.3700, 0.3626, 0.3617, 0.3606, 0.3637]),
    },
}
# fmt: on

# Comparisons against best published baseline for each metric.
# Baseline values are single reported numbers (no per-seed distribution available).
comparisons = [
    ("CaRB (1-1) F1",  per_seed["CaRB (1-1) F1"]["f1"],  0.515, "DualOIE"),
    ("CaRB (1-1) AUC", per_seed["CaRB (1-1) F1"]["auc"],  0.336, "CycleOIE"),
    ("BenchIE F1",     per_seed["BenchIE"]["f1"],          0.340, "ClausIE"),
]

print(f"{'Metric':<18} {'Ours':>6} {'vs':>4} {'Baseline':>9} {'System':<10} {'t':>7} {'p (one-sided)':>14} {'':>5}")
print("-" * 82)
for name, seeds, baseline, system in comparisons:
    t, p_two = ttest_1samp(seeds, baseline)
    p_one = p_two / 2 if t > 0 else 1.0
    sig = "***" if p_one < 0.001 else "**" if p_one < 0.01 else "*" if p_one < 0.05 else "n.s."
    print(f"{name:<18} {seeds.mean()*100:>6.2f} {'vs':>4} {baseline*100:>9.1f} {system:<10} {t:>7.3f} {p_one:>14.4f} {sig:>5}")

print()
print("Significance: * p<0.05  ** p<0.01  *** p<0.001")
print("Test: one-sample one-sided t-test against fixed baseline value (n=10 seeds).")
