"""Import-time identity registry for governed RLRMP data products."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

from feedbax.contracts.graph import AnalysisDataProductRequirement

__all__ = [
    "DataProductIdentity",
    "register_data_product_identity",
    "registered_data_product_identities",
]


@dataclass(frozen=True)
class DataProductIdentity:
    """Registered metadata for one governed data product."""

    role: str
    product_schema_id: str
    product_schema_version: str
    logical_name: str
    requirement_factory: Callable[[], AnalysisDataProductRequirement]
    document_relpath: str


_DATA_PRODUCT_IDENTITIES: dict[str, DataProductIdentity] = {}


def register_data_product_identity(
    *,
    role: str,
    product_schema_id: str,
    product_schema_version: str,
    logical_name: str,
    requirement_factory: Callable[[], AnalysisDataProductRequirement],
    document_relpath: str,
) -> None:
    """Register one data-product identity, failing closed on key collisions."""

    candidate = DataProductIdentity(
        role=role,
        product_schema_id=product_schema_id,
        product_schema_version=product_schema_version,
        logical_name=logical_name,
        requirement_factory=requirement_factory,
        document_relpath=document_relpath,
    )
    for key in ("role", "product_schema_id", "logical_name"):
        existing = _registered_by_key(key, getattr(candidate, key))
        if existing is not None and not _same_identity(existing, candidate):
            raise ValueError(
                "Data product identity collision on "
                f"{key}: existing {_format_identity(existing)}; "
                f"new {_format_identity(candidate)}"
            )
    _DATA_PRODUCT_IDENTITIES.setdefault(candidate.role, candidate)


def registered_data_product_identities() -> Mapping[str, DataProductIdentity]:
    """Return registered data-product identities keyed by role."""

    return MappingProxyType(dict(_DATA_PRODUCT_IDENTITIES))


def _registered_by_key(key: str, value: str) -> DataProductIdentity | None:
    for identity in _DATA_PRODUCT_IDENTITIES.values():
        if getattr(identity, key) == value:
            return identity
    return None


def _same_identity(left: DataProductIdentity, right: DataProductIdentity) -> bool:
    return (
        left.role == right.role
        and left.product_schema_id == right.product_schema_id
        and left.product_schema_version == right.product_schema_version
        and left.logical_name == right.logical_name
        and _factory_key(left.requirement_factory) == _factory_key(right.requirement_factory)
        and left.document_relpath == right.document_relpath
    )


def _factory_key(factory: Callable[[], AnalysisDataProductRequirement]) -> tuple[str, str]:
    return (factory.__module__, factory.__qualname__)


def _format_identity(identity: DataProductIdentity) -> str:
    return (
        f"role={identity.role!r}, "
        f"product_schema_id={identity.product_schema_id!r}, "
        f"logical_name={identity.logical_name!r}, "
        f"document_relpath={identity.document_relpath!r}"
    )
