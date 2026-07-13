# Defaults-ratchet shrink plan

## Baseline

Issue b2562ad's completed intent purge landed as commit `d461adba`. That commit
removed literal fallbacks from the three authored training surfaces guarded by
`REQUIRED_AUTHORING_SCAN_TARGETS` in `tests/test_defaults_ratchet.py`. Those
paths now have two permanent conditions: they remain in scanner scope with zero
fallbacks, and they may not regain allowlist entries.

The remaining `ci/defaults-ratchet-allowlist.toml` inventory is transitional,
not a compatibility surface. Its owner groups remain the source-fix sequence:

1. `9f6d2a5`: seed, timestep, and benchmark fallbacks.
2. `1d59c4d`: training-run spec fallbacks.
3. `cc3a61f`: recipe, report, and training fallbacks.
4. `7cfe941`: calibration and data-product fallbacks.
5. `e1afa46`: clamp gain-normalization diagnostics.
6. `5b3aabe`: catch-all entries that must be assigned to a narrower source-fix
   lane before remediation.

## Retirement rule

Each source-fix commit removes the corresponding `(path, key, literal_repr)`
allowlist entry, or lowers its count by exactly the occurrences removed. An
entry is never renamed, reassigned, or re-added to make a source change pass.
If the source still needs a value, the value moves to its schema-owned params
model; if the fallback is genuinely local result metadata, the owning lane
documents that semantic distinction in the test's narrow exception inventory.

The ratchet continues to reject new or grown occurrences immediately. Stale
entries outside the completed intent-purge baseline remain temporarily readable
only so independent owner lanes can land without a cross-file flag day; their
next source edit must apply the retirement rule above.
