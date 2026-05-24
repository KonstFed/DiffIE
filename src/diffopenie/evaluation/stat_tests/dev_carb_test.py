# Paired one-sided t-test comparing lsoie-ex-2.5k vs lsoie-ex-full on the dev set.
# Paired because both configurations were evaluated at the same 10 seeds (38-47),
# so seed-level correlation is real and pairing removes it from the error term.
# H0: lsoie-ex-2.5k <= lsoie-ex-full; H1: lsoie-ex-2.5k > lsoie-ex-full.
# Goal: confirm that the smaller 2.5k configuration is not statistically worse
# than training on all available curated data, justifying the choice of 2.5k.

import numpy as np
from scipy.stats import ttest_rel

# fmt: off
scores = {
    "CaRB F1": {
        "2500": np.array([0.525, 0.527, 0.522, 0.525, 0.521, 0.527, 0.518, 0.524, 0.524, 0.522]),
        "full": np.array([0.523, 0.527, 0.524, 0.525, 0.524, 0.524, 0.525, 0.528, 0.526, 0.525]),
    },
    "CaRB (1-1) F1": {
        "2500": np.array([0.512, 0.517, 0.512, 0.520, 0.511, 0.521, 0.516, 0.518, 0.512, 0.517]),
        "full": np.array([0.515, 0.517, 0.515, 0.515, 0.516, 0.515, 0.511, 0.514, 0.513, 0.512]),
    },
}
# fmt: on

print(f"{'Metric':<20} {'2500 mean':>10} {'full mean':>10} {'diff':>7} {'t':>7} {'p (one-sided)':>14} {'':>5}")
print("-" * 76)
for name, s in scores.items():
    diff = s["2500"] - s["full"]
    res = ttest_rel(s["2500"], s["full"], alternative="greater")
    p_one = res.pvalue
    sig = "***" if p_one < 0.001 else "**" if p_one < 0.01 else "*" if p_one < 0.05 else "n.s."
    print(
        f"{name:<20} {s['2500'].mean()*100:>10.2f} {s['full'].mean()*100:>10.2f}"
        f" {diff.mean()*100:>7.2f} {res.statistic:>7.3f} {p_one:>14.4f} {sig:>5}"
    )

print()
print("Significance: * p<0.05  ** p<0.01  *** p<0.001")
print("Test: paired one-sided t-test, n=10 seeds, dev set.")
