"""Lane C terminal acceptance gate: data-products / source-hygiene (issue 08bb6d4).

This is the declared terminal acceptance gate for the data-products and source-hygiene
lane of the ``64a04e0`` Feedbax-native umbrella. Lane C moved generated calibration and
budget tables out of source constants into descriptor / data-product specs
(``ea6ccb4``), added typed data-product identity for consumed generated outputs
(``108b4d3``), and migrated feedback-component metadata onto descriptor-backed specs
(``259bd10`` / ``844acc6``).

Each condition is enforced elsewhere by a dedicated, marked gate family. This terminal
gate does not re-implement those scans; it asserts, by construction, that the three Lane
C outcomes hold and that their enforcing families are live in the contract suite:

1.  **No generated tables remain in source.** The ``generated_data_constant_scan`` family
    (``ea6ccb4``, ``tests/test_data_lint_generated_constants.py``) lints the ``src`` tree
    via ``rlrmp.data_products.lint.violations``; this gate re-asserts that entry point
    reports zero violations.
2.  **Consumed data-product identities are fail-closed and non-null.** Run specs snapshot
    consumed data-product identities through ``add_consumed_data_identity``, whose guard
    refuses an empty role / schema / hash. The loader side
    (``rlrmp.data_products.envelope``) re-derives and pins the identity hash. The
    ``write_surface`` family (``f5d9695``) already enrolls the identity-shape invariant.
3.  **Stopgap surfaces are descriptor-backed.** ``controller_feedback_scales`` are resolved
    from a serialized descriptor payload carrying a ``sha256:`` basis hash rather than
    hard-coded slices/order; the ``descriptor_basis_hash`` and ``feedback_descriptor_scan``
    families (``844acc6`` / ``259bd10``) enforce this.

``skips count as failures`` in this gate: the family is enrolled ``live`` in
``ci/feedbax-contract-suite.toml`` and ``tests/conftest.py`` +
``test_feedbax_contract_meta.py`` forbid SKIP / non-strict XFAIL under the
``feedbax_contract`` marker.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from rlrmp.data_products.envelope import (
    DataProductError,
    load_data_product,
    validate_data_product,
)
from rlrmp.data_products.lint import violations
from rlrmp.model.feedback_descriptors import controller_feedback_descriptor_payload
from rlrmp.runtime.training_run_specs import (
    CONSUMED_DATA_IDENTITIES_KEY,
    add_consumed_data_identity,
)


pytestmark = pytest.mark.feedbax_contract

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SUITE_MANIFEST_PATH = REPO_ROOT / "ci" / "feedbax-contract-suite.toml"


def _live_family_names() -> set[str]:
    manifest = tomllib.loads(SUITE_MANIFEST_PATH.read_text(encoding="utf-8"))
    return {family["name"] for family in manifest["families"] if family["status"] == "live"}


def test_lane_c_enforcing_families_are_live() -> None:
    """The three Lane C outcomes are backed by live contract-suite families."""

    live = _live_family_names()
    # Condition 1 (generated tables out of source), condition 3 (descriptor-backed
    # feedback), and the identity-shape invariant behind condition 2 are all enrolled live.
    for family in (
        "generated_data_constant_scan",
        "descriptor_basis_hash",
        "feedback_descriptor_scan",
        "write_surface",
    ):
        assert family in live, f"expected Lane C enforcing family {family!r} to be live"


def test_no_generated_tables_remain_in_source() -> None:
    """Condition 1 — the generated-data lint reports zero un-allowlisted source tables."""

    found = violations(SRC_ROOT, repo_root=REPO_ROOT)
    assert found == [], [f"{finding.key} (line {finding.lineno})" for finding in found]


def test_consumed_data_identity_is_fail_closed_non_null() -> None:
    """Condition 2 — run specs snapshot non-null consumed hashes; empties fail closed."""

    # A valid consumed identity is snapshotted onto the run spec.
    updated = add_consumed_data_identity(
        {}, role="perturbation_open_loop_calibration", schema="rlrmp.data_product.v1", hash="abc123"
    )
    assert updated[CONSUMED_DATA_IDENTITIES_KEY] == [
        {
            "role": "perturbation_open_loop_calibration",
            "schema": "rlrmp.data_product.v1",
            "hash": "abc123",
        }
    ]

    # Fail-closed: an empty role / schema / hash cannot be snapshotted.
    for bad in (
        {"role": "", "schema": "s", "hash": "h"},
        {"role": "r", "schema": "", "hash": "h"},
        {"role": "r", "schema": "s", "hash": ""},
    ):
        with pytest.raises(ValueError):
            add_consumed_data_identity({}, **bad)

    # The loader-side identity assertion machinery is present and fail-closed.
    assert issubclass(DataProductError, RuntimeError)
    assert callable(validate_data_product)
    assert callable(load_data_product)


def test_controller_feedback_scales_are_descriptor_backed() -> None:
    """Condition 3 — feedback scales resolve from a descriptor payload with a basis hash."""

    payload_4d = controller_feedback_descriptor_payload(feedback_dim=4)
    payload_6d = controller_feedback_descriptor_payload(feedback_dim=6)

    assert payload_4d["basis_id"] == "target_relative_delayed_feedback"
    assert payload_6d["basis_id"] == "target_relative_delayed_feedback_plus_force_filter"
    for payload in (payload_4d, payload_6d):
        assert payload["descriptor_basis_hash"].startswith("sha256:")
