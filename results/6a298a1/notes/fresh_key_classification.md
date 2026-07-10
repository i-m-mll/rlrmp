# Fresh-key classification

The strengthened D3/D4 scanner added 31 ratchet keys that were absent from the older
`b6136cc` inventory. They were classified from their live definitions and call sites before
migration.

## Historical/spec migration

- `results/7c1f7ed/scripts/materialize_delayed_sisu_velocity_profiles.py::main::argv_rows`
  is a historical argv-encoded condition row. It must be materialized as a tracked
  `results/7c1f7ed` run/evaluation spec before the literal row is removed.

## Existing owning schema defaults

These findings are typed config/params models already owning their field defaults. Their
destination is the class itself, recorded through the curated schema-default exception:

- `analysis/math/hinf_riccati.py::CostSpec`
- `analysis/math/linear_equivalence_certificate.py::CertificateConfig`
- `analysis/math/linear_round_trip.py::LinearTrainingConfig`
- `analysis/math/linear_round_trip.py::TeacherFitConfig`
- `analysis/math/output_feedback.py::OutputFeedbackConfig`
- `analysis/pipelines/cs_stochastic_phase3.py::Phase3StochasticConfig`
- `analysis/pipelines/gru_checkpoint_selection.py::DelayedReachEvalBankSpec`
- `analysis/pipelines/gru_steady_state_perturbation_bank.py::SteadyStatePerturbationBankConfig`
- `analysis/robustness_margin.py::RobustnessMarginParams`
- `cloud/modal_runner.py::NominalGruRunConfig`
- `model/stochastic_runtime.py::StochasticRuntimeConfig`

## Config-tier ownership exclusion

The following 11 keys are part of the unfinished unified training-config/pre-native
retirement surface owned by `cd137d8`. They remain live and named under that issue-specific
curated exception:

- `train/closed_loop_distillation.py::ClosedLoopLossWeights`
- `train/cs_nominal_gru.py::AdaptiveEpsilonState`
- `train/cs_nominal_gru.py::CsNominalGruConfig`
- `train/cs_nominal_gru.py::build_hps`
- `train/cs_perturbation_training.py::BroadFullStateEpsilonTrainingConfig`
- `train/cs_perturbation_training.py::FixedTargetPerturbationTrainingConfig`
- `train/cs_perturbation_training.py::PgdFullStateEpsilonTrainingConfig`
- `train/cs_perturbation_training.py::PolicyFullStateEpsilonTrainingConfig`
- `train/cs_perturbation_training.py::TargetRelativeMultiTargetTrainingConfig`
- `train/minimax.py::MinimaxConfig`
- `train/minimax.py::build_hps`

## Explicit user hold

- `fb_response.py::InstantFBResponse`
- `model/feedbax_graph.py::_linear_controller_params`

Both objects are in the `e04bd36` confirmed-dead portfolio. The user hold prohibits their
deletion or modification, so they remain as explicit curated exceptions.

## New migration work

- `analysis/pipelines/sisu_spectrum_diagnostics.py::DEFAULT_REFERENCE_SAMPLES` is an
  analysis parameter and moves to the registered analysis params/spec surface.
- `model/cs_lss_gru.py::_population_structure_params` is a reusable model-construction
  preset and moves to a registered model preset/template.
- `model/feedbax_graph.py::DEFAULT_GRAPH_COMPONENT_SEED` is a graph-construction default
  and moves to the owning graph-spec/preset surface.
- `model/feedbax_graph.py::_migrate_legacy_plant_process_force_noise_params` contains
  compatibility migration defaults; the values move to the versioned migration/schema
  surface before the function literal disappears.
- `model/feedbax_graph.py::_population_structure_params` and
  `_runtime_population_structure_params` are reusable graph/model parameter bundles and
  move to the registered graph/model preset surface.

Co-Authored-By: Codex (GPT-5) <codex@openai.com>
