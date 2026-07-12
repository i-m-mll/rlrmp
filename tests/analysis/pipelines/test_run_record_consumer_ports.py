from __future__ import annotations

import inspect
import re

import rlrmp.analysis.gru_standard_certificate as cs_standard
import rlrmp.analysis.pipelines.gru_broad_epsilon_attribution as broad_epsilon
from rlrmp.eval import checkpoint_selection
import rlrmp.eval.trial_inputs as evaluation_run_inputs
import rlrmp.analysis.pipelines.objective_comparator as objective_comparator


CONSUMERS = (
    cs_standard,
    evaluation_run_inputs,
    checkpoint_selection,
    objective_comparator,
    broad_epsilon,
)


def test_governed_run_record_consumers_use_resolver() -> None:
    forbidden = re.compile(
        r"(run_spec_path|tracked_run_spec_path)\([^)]*\)\.read_text|"
        r"glob\([\"']\*/run\.json[\"']\)|"
        r"runs[\"']\s*/\s*run_id\s*/\s*[\"']run\.json"
    )
    for module in CONSUMERS:
        source = inspect.getsource(module)
        assert "resolve_run_record" in source, module.__name__
        assert not forbidden.search(source), module.__name__
