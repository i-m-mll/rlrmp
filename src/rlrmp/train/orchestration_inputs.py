"""Immutable checkpoint-transaction input binding for orchestration."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from feedbax.orchestration.assembly import AssemblyInputDeclaration
from feedbax.orchestration.bundle import ImmutableInputDigest, ImmutableInputIdentity


LOCATOR_SCHEME = "checkpoint-transaction"


def checkpoint_transaction_locator(root: Path, *, transaction_id: str, manifest_sha256: str) -> str:
    """Build the immutable locator recorded in a run assembly request."""
    return (
        f"{LOCATOR_SCHEME}://{quote(str(root.resolve()), safe='')}"
        f"/{quote(transaction_id, safe='')}?manifest_sha256={manifest_sha256}"
    )


class CheckpointTransactionInputResolver:
    """Resolve recorded checkpoint custody without minting a new identity."""

    def __call__(self, declaration: AssemblyInputDeclaration) -> ImmutableInputIdentity:
        if "latest.json" in declaration.locator:
            raise ValueError("checkpoint input locator must not reference mutable latest.json")
        parsed = urlparse(declaration.locator)
        if parsed.scheme != LOCATOR_SCHEME:
            raise ValueError(f"unsupported checkpoint input locator: {declaration.locator!r}")
        root = Path(unquote(parsed.netloc))
        transaction_id = unquote(parsed.path.lstrip("/"))
        expected = parse_qs(parsed.query).get("manifest_sha256", [None])[0]
        if not transaction_id or not expected:
            raise ValueError("checkpoint locator requires transaction id and manifest_sha256")
        manifest = root / "transactions" / transaction_id / "manifest.json"
        if not manifest.is_file():
            raise ValueError(f"checkpoint transaction is missing: {transaction_id}")
        actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                "checkpoint transaction manifest digest mismatch: "
                f"expected={expected} actual={actual}"
            )
        return ImmutableInputIdentity(
            role=declaration.role,
            kind=declaration.kind,
            identifier=f"{LOCATOR_SCHEME}:{transaction_id}",
            digest=ImmutableInputDigest(value=actual),
            schema_id=declaration.schema_id,
            schema_version=declaration.schema_version,
        )


__all__ = ["CheckpointTransactionInputResolver", "checkpoint_transaction_locator"]
