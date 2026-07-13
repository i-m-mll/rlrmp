# Cross-architecture certificate agreement across evaluation lenses

This experiment family asks whether nominal-versus-robust conclusions agree across a
static-gain linear controller, an augmented-linear recurrent controller, and a nonlinear
GRU when the plant, task, budget, seed, and evaluation disturbances are matched. It is a
local engineering-smoke test only: it cannot answer the scientific question. The frozen
cohort has six intended training rows and four post-training evaluation lenses. The
low-level `static_gain`, `augmented_linear`, and `empirical_nonlinear` certificate
components, explicit `not_applicable` semantics, and custody-routed certificate report
renderer are already present. Execution is instead blocked by two narrower road gaps:
linked issue `427d0d8` owns content-pinned static-linear and linear-recurrent training
bases, while the grouped analysis road still lacks a registered manifest-native adapter
that preserves architecture/mode, nominal-versus-robust distribution, and lens across
heterogeneous `EvaluationRunManifest` inputs. The packet refuses inline bases, legacy
payloads, hand joins, and GRU/static-gain coercion. Exact intended identities and blockers
are recorded in `runs/cohort.intent.json` and `analysis/cross_lens/spec.json`.
