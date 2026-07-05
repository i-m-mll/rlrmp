<!-- AUTO-GENERATED: adaptive_damage_lambda -->
## Adaptive damage and lambda

Actual damage uses `adaptive_epsilon_adaptive_update_damage_raw`, averaged across the five replicate columns at each adaptive diagnostics sample. The plotted target uses `adaptive_epsilon_target_damage`, the smoother feedback signal uses `adaptive_epsilon_damage_ema`, and adaptive lambda uses the post-update `adaptive_epsilon_lambda_value`. The figure uses the recorded `adaptive_epsilon_global_batch` x-axis.

| row | records | batch span | intended endpoint | damage near intended | damage final | lambda near intended | lambda final | damage range | lambda range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short_3500to1000 | 7500 | 12000-19499 | 15500 | 1110.559 | 1050.215 | 5.54381e+06 | 5.54381e+06 | 1336.451 | 3.59515e+08 |
| medium_3500to1000 | 7000 | 12000-18999 | 17250 | 1086.853 | 1056.725 | 717147 | 717147 | 1389.124 | 3.96001e+08 |

Overrun context: short overran beyond intended 15500 and records extend to global batch 19499 (19500 completed batches); medium continued past its intended anneal endpoint and was stopped at 19000 completed batches.

No scientific verdict is inferred here beyond the plotted diagnostics.

Figure: `results/91a090c/figures/adaptive_damage_lambda/figure.html`
Summary JSON: `results/91a090c/figures/adaptive_damage_lambda/summary.json`
<!-- /AUTO-GENERATED -->
