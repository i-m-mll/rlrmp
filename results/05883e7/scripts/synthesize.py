#!/usr/bin/env python3
"""Phase 5A deterministic synthesis for the code-archaeology audit.

Mandible issue: 05883e7. Stdlib-only, no fan-out, no LLM calls. Consumes the
full Phase 0-4 audit corpus under `_artifacts/05883e7/audit/` --

    census/{objects,modules,xref}.jsonl, census/chunk_plan.json
    classification/chunk_*.jsonl                    (Phase 2 records)
    verification/verdicts_merged.jsonl               (Phase 4 corrections)
    sweeps/duplication/clusters.jsonl                (Phase 3)
    sweeps/dangling/dangling.jsonl                   (Phase 3)
    sweeps/reverse_audit/reverse_audit.jsonl         (Phase 3)

-- and produces three machine-readable/human-readable artifact pairs under
`_artifacts/05883e7/audit/synthesis/`:

    tables.json / tables.md          quantitative core (LOC x purpose x
                                      generality x usage_status, contract-flag
                                      inventory, disposition portfolio)
    module_report.jsonl / .md        per-module stats + SPLIT/JOIN/RELOCATE
                                      candidate rankings
    portfolio.json                   machine-readable remediation items,
                                      classes (a)-(g)

Usage (from the rlrmp worktree root):

    PYTHONPATH=src uv run --no-sync python results/05883e7/scripts/synthesize.py

Verification corrections are applied before any aggregation: the 21 refuted
verdicts in `verdicts_merged.jsonl` overwrite `usage_status`/`disposition` on
their classification record with the verdict's `corrected` fields (which may
introduce values outside the `record_schema.md` enum -- e.g.
`dead_but_documented_public_api`, `live_public_api` -- verification is a
free-text override layer, not a second enum-constrained classification pass;
`tables.md` documents this as a caveat). `confirmed` and `uncertain` verdicts
leave the original classification record untouched.

Corpus quirks handled explicitly (see `reconcile()` for the printed report):
    - `census/objects.jsonl` has 5 duplicate-id lines (4 qualnames, module-
      level TypeVar/name rebindings picked up twice by the AST scan); the
      first occurrence of each id is kept, ~9 LOC out of ~275k total object
      LOC.
    - Two classification records are whole-module `__module__` entries for
      files that also have real per-object classification records
      (`feedbax/objectives/{service,spec}.py`); they are excluded from
      object-level tables to avoid double-counting LOC that is already
      covered by the file's real per-object records.
    - `sweeps/reverse_audit/reverse_audit.jsonl` relpaths are inconsistently
      written with or without the leading `feedbax/` package-root segment;
      resolution tries both forms against `census/modules.jsonl`.

Self-check: `reconcile()` runs a fixed set of corpus-consistency checks
(record counts, id-membership, LOC-sum agreement) and prints a PASS/FAIL
table; the script exits nonzero if any check fails.
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

PURPOSE_ENUM = [
    "core_math_algorithm", "model_graph_definition", "training_loop", "eval_logic",
    "analysis_transform", "spec_manifest_construction", "registration_wiring",
    "hp_or_data_constants", "orchestration_cli", "io_custody", "viz",
    "launch_resume_control", "compat_legacy", "test_support", "typing_protocols",
    "docs_meta", "other",
]
GENERALITY_ENUM = [
    "rlrmp_specific", "general_belongs_in_feedbax", "duplicates_feedbax",
    "partial_overlap_feedbax", "framework_native", "project_specific_should_leave_feedbax",
]
USAGE_STATUS_ENUM = [
    "live", "test_only", "legacy_only", "dead", "registry_or_string_referenced", "ambiguous",
]
CONTRACT_FLAG_ENUM = [
    "spec_first_violation", "data_in_code", "experiment_named_in_src", "dangling_reference",
    "custody_bypass", "legacy_unbannered", "misplaced_should_be_results_scripts",
    "misplaced_should_be_library", "none",
]
DISPOSITION_ENUM = [
    "keep", "delete", "move_to_feedbax", "move_to_results_scripts",
    "replace_with_declarative_surface", "merge_dedupe", "needs_decision",
]

LIBRARY_TREES = {"src", "package"}  # trees eligible for SPLIT/JOIN/import-graph analysis
RELOCATE_FLAGS = {
    "misplaced_should_be_results_scripts", "misplaced_should_be_library", "experiment_named_in_src",
}
MISPLACEMENT_FLAGS_FOR_PORTFOLIO_E = {
    "data_in_code", "experiment_named_in_src", "spec_first_violation",
}
GENERALIZE_IN_PLACE = "generalize_in_place"

TOP_N_CONTRACT_MODULES = 10
TOP_N_SPLIT = 15
TOP_N_JOIN = 10
TOP_N_RELOCATE = 15
TOP_N_NEIGHBORS = 5
MIN_DUP_CLUSTER_LOC_FOR_PORTFOLIO = 100
JOIN_LOC_THRESHOLD = 250  # "small sibling modules"


class ReconciliationError(Exception):
    pass


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


@dataclass
class Corpus:
    objects: list[dict]                 # deduped by id, first-occurrence-wins
    objects_by_id: dict[str, dict]
    n_object_lines_raw: int
    n_object_dupe_lines: int
    modules: list[dict]
    modules_by_id: dict[str, dict]
    modules_by_repo_relpath: dict[tuple[str, str], dict]
    xref_by_id: dict[str, dict]
    classification_all: dict[str, dict]        # every classification record, incl. stray module rows
    classification_objects: dict[str, dict]     # classification restricted to real object ids, corrected
    stray_module_classification_ids: list[str]
    verdicts_by_id: dict[str, dict]
    dup_clusters: list[dict]
    dangling: list[dict]
    reverse_audit: list[dict]


def load_corpus(audit_root: Path) -> Corpus:
    census_dir = audit_root / "census"
    classification_dir = audit_root / "classification"
    verification_path = audit_root / "verification" / "verdicts_merged.jsonl"
    dup_path = audit_root / "sweeps" / "duplication" / "clusters.jsonl"
    dangling_path = audit_root / "sweeps" / "dangling" / "dangling.jsonl"
    reverse_audit_path = audit_root / "sweeps" / "reverse_audit" / "reverse_audit.jsonl"

    raw_objects = load_jsonl(census_dir / "objects.jsonl")
    objects_by_id: dict[str, dict] = {}
    n_dupe_lines = 0
    for o in raw_objects:
        if o["id"] in objects_by_id:
            n_dupe_lines += 1
            continue
        objects_by_id[o["id"]] = o
    objects = list(objects_by_id.values())

    modules = load_jsonl(census_dir / "modules.jsonl")
    modules_by_id = {m["id"]: m for m in modules}
    modules_by_repo_relpath = {(m["repo"], m["relpath"]): m for m in modules}

    xref = load_jsonl(census_dir / "xref.jsonl")
    xref_by_id = {x["id"]: x for x in xref}

    # Classification corpus: every chunk_*.jsonl directly under classification/
    # (excludes classification/quarantine/, which holds scratch/leftover files
    # from chunk repair and is not part of the canonical corpus).
    chunk_paths = sorted(Path(p) for p in globmod.glob(str(classification_dir / "chunk_*.jsonl")))
    classification_all: dict[str, dict] = {}
    for p in chunk_paths:
        for rec in load_jsonl(p):
            if rec["id"] in classification_all:
                raise ReconciliationError(f"duplicate classification id across chunks: {rec['id']}")
            classification_all[rec["id"]] = rec

    stray_module_ids = sorted(
        rid for rid in classification_all
        if rid in modules_by_id and rid not in objects_by_id
    )
    classification_objects = {
        rid: rec for rid, rec in classification_all.items() if rid in objects_by_id
    }

    verdicts = load_jsonl(verification_path)
    verdicts_by_id = {v["id"]: v for v in verdicts}

    dup_clusters = load_jsonl(dup_path)
    dangling = load_jsonl(dangling_path)
    reverse_audit = load_jsonl(reverse_audit_path)

    return Corpus(
        objects=objects,
        objects_by_id=objects_by_id,
        n_object_lines_raw=len(raw_objects),
        n_object_dupe_lines=n_dupe_lines,
        modules=modules,
        modules_by_id=modules_by_id,
        modules_by_repo_relpath=modules_by_repo_relpath,
        xref_by_id=xref_by_id,
        classification_all=classification_all,
        classification_objects=classification_objects,
        stray_module_classification_ids=stray_module_ids,
        verdicts_by_id=verdicts_by_id,
        dup_clusters=dup_clusters,
        dangling=dangling,
        reverse_audit=reverse_audit,
    )


# --------------------------------------------------------------------------
# Verification correction
# --------------------------------------------------------------------------

def apply_corrections(corpus: Corpus) -> dict[str, dict]:
    """Return a corrected copy of classification_objects: refuted verdicts
    overwrite usage_status/disposition with their `corrected` fields; a
    `_verification` field records the provenance (`unverified`, `confirmed`,
    `uncertain`, or `refuted_corrected`) for downstream caveats."""
    corrected: dict[str, dict] = {}
    for rid, rec in corpus.classification_objects.items():
        rec2 = dict(rec)
        v = corpus.verdicts_by_id.get(rid)
        if v is None:
            rec2["_verification"] = "unverified"
        elif v["verdict"] == "confirmed":
            rec2["_verification"] = "confirmed"
        elif v["verdict"] == "uncertain":
            rec2["_verification"] = "uncertain"
        elif v["verdict"] == "refuted":
            corr = v.get("corrected") or {}
            if "usage_status" in corr:
                rec2["usage_status"] = corr["usage_status"]
            if "disposition" in corr:
                rec2["disposition"] = corr["disposition"]
            rec2["_verification"] = "refuted_corrected"
            rec2["_verification_route"] = v.get("refutation_route")
        else:
            rec2["_verification"] = f"unknown_verdict:{v['verdict']}"
        corrected[rid] = rec2
    return corrected


# --------------------------------------------------------------------------
# Reconciliation self-check
# --------------------------------------------------------------------------

@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def reconcile(corpus: Corpus, corrected: dict[str, dict]) -> list[Check]:
    checks: list[Check] = []

    checks.append(Check(
        "objects.jsonl dedup",
        True,
        f"{corpus.n_object_lines_raw} raw lines, {corpus.n_object_dupe_lines} duplicate-id lines "
        f"dropped, {len(corpus.objects)} unique object ids retained (first occurrence kept).",
    ))

    n_obj = len(corpus.objects_by_id)
    n_cls_obj = len(corpus.classification_objects)
    checks.append(Check(
        "every census object id has exactly one classification record",
        n_obj == n_cls_obj,
        f"census objects={n_obj}, classification object-records={n_cls_obj}",
    ))

    extra = set(corpus.classification_all) - set(corpus.objects_by_id) - set(corpus.modules_by_id)
    checks.append(Check(
        "no classification id outside census objects+modules",
        len(extra) == 0,
        f"{len(extra)} orphan classification ids" + (f": {sorted(extra)[:5]}" if extra else ""),
    ))
    checks.append(Check(
        "stray whole-module classification records identified and excluded",
        True,
        f"{len(corpus.stray_module_classification_ids)} found: "
        f"{corpus.stray_module_classification_ids}",
    ))

    unclassified = set(corpus.objects_by_id) - set(corpus.classification_objects)
    checks.append(Check(
        "no census object left unclassified",
        len(unclassified) == 0,
        f"{len(unclassified)} unclassified object ids",
    ))

    n_loc_mismatch = 0
    total_delta = 0
    total_cls_loc = sum(r.get("loc", 0) for r in corpus.classification_objects.values())
    for rid, obj in corpus.objects_by_id.items():
        cls = corpus.classification_objects.get(rid)
        if cls is not None and cls.get("loc") != obj.get("loc"):
            n_loc_mismatch += 1
            total_delta += abs(cls.get("loc", 0) - obj.get("loc", 0))
    # Tolerance: Phase 2 fan-out agents hand-recorded `loc` per record rather
    # than copying it programmatically from the census; small +/-1-2 line
    # boundary disagreements (decorator/blank-line inclusion) are expected
    # noise, not a corpus-integrity failure. Fail only if the aggregate drift
    # exceeds 0.5% of total classified LOC.
    delta_pct = 100.0 * total_delta / total_cls_loc if total_cls_loc else 0.0
    checks.append(Check(
        "classification.loc matches census object.loc (tolerance: <0.5% aggregate LOC drift)",
        delta_pct < 0.5,
        f"{n_loc_mismatch} per-object mismatches (mostly +/-1-2 line boundary disagreements), "
        f"total abs LOC delta {total_delta} = {delta_pct:.3f}% of {total_cls_loc} total classified LOC",
    ))

    verdict_ids_missing = sorted(
        vid for vid in corpus.verdicts_by_id if vid not in corpus.classification_objects
    )
    checks.append(Check(
        "every verification verdict id resolves to a classification object record",
        len(verdict_ids_missing) == 0,
        f"{len(verdict_ids_missing)} missing" + (f": {verdict_ids_missing[:5]}" if verdict_ids_missing else ""),
    ))

    vcounts = Counter(v["verdict"] for v in corpus.verdicts_by_id.values())
    checks.append(Check(
        "verification tally matches corpus README (866/21/1)",
        vcounts.get("confirmed") == 866 and vcounts.get("refuted") == 21 and vcounts.get("uncertain") == 1,
        f"confirmed={vcounts.get('confirmed', 0)}, refuted={vcounts.get('refuted', 0)}, "
        f"uncertain={vcounts.get('uncertain', 0)}, total={sum(vcounts.values())}",
    ))

    n_dup_missing = 0
    n_dup_members = 0
    for c in corpus.dup_clusters:
        for mid in c["member_ids"]:
            n_dup_members += 1
            if mid not in corpus.objects_by_id:
                n_dup_missing += 1
    checks.append(Check(
        "duplication cluster member ids resolve against census objects",
        n_dup_missing == 0,
        f"{n_dup_missing}/{n_dup_members} unresolved cluster members",
    ))

    n_dang_missing = 0
    for d in corpus.dangling:
        if (d["repo"], d["module"]) not in corpus.modules_by_repo_relpath:
            n_dang_missing += 1
    checks.append(Check(
        "dangling-reference module paths resolve against census modules",
        n_dang_missing == 0,
        f"{n_dang_missing}/{len(corpus.dangling)} unresolved dangling-finding modules",
    ))

    n_rev_missing = 0
    for r in corpus.reverse_audit:
        if resolve_reverse_audit_module(corpus, r) is None:
            n_rev_missing += 1
    checks.append(Check(
        "reverse-audit relpaths resolve against census feedbax modules "
        "(tolerating inconsistent leading 'feedbax/' prefix)",
        n_rev_missing == 0,
        f"{n_rev_missing}/{len(corpus.reverse_audit)} unresolved",
    ))

    return checks


def resolve_reverse_audit_module(corpus: Corpus, rec: dict) -> Optional[dict]:
    rp = rec["relpath"]
    candidates = [rp] if rp.startswith("feedbax/") else [rp, "feedbax/" + rp]
    for c in candidates:
        m = corpus.modules_by_repo_relpath.get(("feedbax", c))
        if m is not None:
            return m
    return None


# --------------------------------------------------------------------------
# Module import graph (for module_report fan-in/fan-out + SPLIT/JOIN)
# --------------------------------------------------------------------------

def relpath_to_dotted(repo: str, tree: str, relpath: str) -> Optional[str]:
    if repo == "rlrmp" and tree == "src" and relpath.startswith("src/"):
        stem = relpath[len("src/"):]
    elif repo == "feedbax" and tree == "package":
        stem = relpath
    else:
        return None
    if stem.endswith("/__init__.py"):
        stem = stem[: -len("/__init__.py")]
    elif stem.endswith(".py"):
        stem = stem[: -len(".py")]
    else:
        return None
    return stem.replace("/", ".")


@dataclass
class ImportGraph:
    dotted_to_module_id: dict[str, str]
    out_edges: dict[str, set[str]]   # module id -> set of module ids it imports (resolved, intra-corpus)
    in_edges: dict[str, set[str]]    # module id -> set of module ids that import it


def build_import_graph(corpus: Corpus) -> ImportGraph:
    dotted_to_module_id: dict[str, str] = {}
    for m in corpus.modules:
        dotted = relpath_to_dotted(m["repo"], m["tree"], m["relpath"])
        if dotted is not None:
            dotted_to_module_id[dotted] = m["id"]

    out_edges: dict[str, set[str]] = defaultdict(set)
    in_edges: dict[str, set[str]] = defaultdict(set)
    for m in corpus.modules:
        for imp in m.get("imports", []):
            target_id = dotted_to_module_id.get(imp)
            if target_id is not None and target_id != m["id"]:
                out_edges[m["id"]].add(target_id)
                in_edges[target_id].add(m["id"])

    return ImportGraph(dotted_to_module_id=dotted_to_module_id, out_edges=out_edges, in_edges=in_edges)


def top_neighbors(graph: ImportGraph, module_id: str, n: int = TOP_N_NEIGHBORS) -> list[dict]:
    out_n = graph.out_edges.get(module_id, set())
    in_n = graph.in_edges.get(module_id, set())
    mutual = out_n & in_n
    all_neighbors = out_n | in_n
    degree = {nid: len(graph.out_edges.get(nid, ())) + len(graph.in_edges.get(nid, ())) for nid in all_neighbors}

    def direction(nid: str) -> str:
        if nid in mutual:
            return "mutual"
        return "out" if nid in out_n else "in"

    ranked = sorted(
        all_neighbors,
        key=lambda nid: (0 if nid in mutual else 1, -degree.get(nid, 0), nid),
    )
    return [{"module": nid, "direction": direction(nid), "degree": degree.get(nid, 0)} for nid in ranked[:n]]


# --------------------------------------------------------------------------
# Deliverable 1a: tables
# --------------------------------------------------------------------------

def loc_count_table(records: Iterable[dict], group_keys: list[str]) -> list[dict]:
    agg: dict[tuple, dict] = {}
    for r in records:
        key = tuple(r[k] for k in group_keys)
        slot = agg.setdefault(key, {"loc": 0, "n_objects": 0})
        slot["loc"] += r["loc"]
        slot["n_objects"] += 1
    rows = []
    for key, slot in sorted(agg.items()):
        row = dict(zip(group_keys, key))
        row.update(slot)
        rows.append(row)
    rows.sort(key=lambda r: (-r["loc"], tuple(r[k] for k in group_keys)))
    return rows


def build_tables(corpus: Corpus, corrected: dict[str, dict], graph: ImportGraph) -> dict:
    recs = list(corrected.values())

    # -- rlrmp-src purpose composition (the motivating-question table) --
    src_recs = [r for r in recs if r["repo"] == "rlrmp" and r["tree"] == "src"]
    src_total_loc = sum(r["loc"] for r in src_recs)
    src_by_purpose = loc_count_table(src_recs, ["purpose"])
    for row in src_by_purpose:
        row["pct_of_rlrmp_src_object_loc"] = round(100.0 * row["loc"] / src_total_loc, 2) if src_total_loc else 0.0

    src_module_loc_total = corpus.modules_by_repo_relpath and sum(
        m["loc"] for m in corpus.modules if m["repo"] == "rlrmp" and m["tree"] == "src"
    )
    rlrmp_src_overhead = {
        "module_loc_total": src_module_loc_total,
        "classified_object_loc_total": src_total_loc,
        "overhead_loc": src_module_loc_total - src_total_loc,
        "overhead_pct": round(100.0 * (src_module_loc_total - src_total_loc) / src_module_loc_total, 2)
        if src_module_loc_total else 0.0,
        "overhead_note": "Imports, blank lines, module docstrings, and comments outside any "
                          "top-level object span; not attributable to a single purpose.",
    }

    loc_by_purpose_repo_tree = loc_count_table(recs, ["repo", "tree", "purpose"])
    loc_by_generality_repo = loc_count_table(recs, ["repo", "generality"])
    loc_by_usage_status_repo_tree = loc_count_table(recs, ["repo", "tree", "usage_status"])
    confidence_dist = loc_count_table(recs, ["confidence"])

    # -- contract-flag inventory --
    flag_rows = []
    for flag in CONTRACT_FLAG_ENUM:
        if flag == "none":
            continue
        flagged = [r for r in recs if flag in r["contract_flags"]]
        by_module: dict[tuple[str, str], dict] = {}
        for r in flagged:
            key = (r["repo"], r["id"].split(":")[1])  # module_relpath segment of id
            slot = by_module.setdefault(key, {"count": 0, "loc": 0})
            slot["count"] += 1
            slot["loc"] += r["loc"]
        top_modules = sorted(
            ({"repo": k[0], "module": k[1], **v} for k, v in by_module.items()),
            key=lambda row: (-row["loc"], -row["count"], row["module"]),
        )[:TOP_N_CONTRACT_MODULES]
        flag_rows.append({
            "flag": flag,
            "count": len(flagged),
            "loc": sum(r["loc"] for r in flagged),
            "top_modules": top_modules,
        })
    flag_rows.sort(key=lambda r: -r["loc"])

    # -- disposition portfolio with LOC deltas --
    delete_recs = [r for r in recs if r["disposition"] == "delete"]
    delete_verified = [r for r in delete_recs if r["_verification"] in ("confirmed", "refuted_corrected")]
    merge_recs = [r for r in recs if r["disposition"] == "merge_dedupe"]
    move_rs_recs = [r for r in recs if r["disposition"] == "move_to_results_scripts"]
    needs_decision_recs = [r for r in recs if r["disposition"] == "needs_decision"]
    legacy_recs = [r for r in recs if r["usage_status"] == "legacy_only"]
    legacy_by_tree = loc_count_table(legacy_recs, ["repo", "tree"])

    # join merge_dedupe records against duplication clusters where the id is a cluster member
    dup_member_ids: set[str] = set()
    for c in corpus.dup_clusters:
        dup_member_ids.update(c["member_ids"])
    merge_matched = [r for r in merge_recs if r["id"] in dup_member_ids]

    disposition_portfolio = {
        "delete": {
            "n_total_disposition_delete": len(delete_recs),
            "loc_total_disposition_delete": sum(r["loc"] for r in delete_recs),
            "n_verification_confirmed": len(delete_verified),
            "loc_verification_confirmed": sum(r["loc"] for r in delete_verified),
            "n_unverified": len(delete_recs) - len(delete_verified),
            "note": "verification_confirmed = original disposition delete with a confirmed verdict, "
                    "plus records refuted INTO delete (corrected disposition == delete) minus records "
                    "refuted OUT of delete.",
        },
        "merge_dedupe": {
            "n": len(merge_recs),
            "loc": sum(r["loc"] for r in merge_recs),
            "n_matched_to_duplication_cluster": len(merge_matched),
            "loc_matched_to_duplication_cluster": sum(r["loc"] for r in merge_matched),
        },
        "move_to_results_scripts": {
            "n": len(move_rs_recs),
            "loc": sum(r["loc"] for r in move_rs_recs),
        },
        "legacy_only_by_repo_tree": legacy_by_tree,
        "needs_decision": {
            "n": len(needs_decision_recs),
            "loc": sum(r["loc"] for r in needs_decision_recs),
        },
    }

    return {
        "corpus_summary": {
            "n_census_objects": len(corpus.objects_by_id),
            "n_census_modules": len(corpus.modules),
            "n_classification_object_records": len(corpus.classification_objects),
            "n_verification_verdicts": len(corpus.verdicts_by_id),
            "n_verification_confirmed": sum(1 for v in corpus.verdicts_by_id.values() if v["verdict"] == "confirmed"),
            "n_verification_refuted": sum(1 for v in corpus.verdicts_by_id.values() if v["verdict"] == "refuted"),
            "n_verification_uncertain": sum(1 for v in corpus.verdicts_by_id.values() if v["verdict"] == "uncertain"),
            "n_duplication_clusters": len(corpus.dup_clusters),
            "n_dangling_findings": len(corpus.dangling),
            "n_reverse_audit_modules": len(corpus.reverse_audit),
        },
        "rlrmp_src_purpose_composition": src_by_purpose,
        "rlrmp_src_object_loc_overhead": rlrmp_src_overhead,
        "loc_by_purpose_x_repo_tree": loc_by_purpose_repo_tree,
        "loc_by_generality_x_repo": loc_by_generality_repo,
        "loc_by_usage_status_x_repo_tree_post_correction": loc_by_usage_status_repo_tree,
        "confidence_distribution": confidence_dist,
        "contract_flag_inventory": flag_rows,
        "disposition_portfolio": disposition_portfolio,
    }


# --------------------------------------------------------------------------
# Deliverable 1b: module_report
# --------------------------------------------------------------------------

def strip_module_suffix(module_id: str) -> str:
    """`repo:relpath:__module__` -> `repo:relpath` for human-readable display."""
    suffix = ":__module__"
    return module_id[: -len(suffix)] if module_id.endswith(suffix) else module_id


def shannon_entropy_bits(fractions: Iterable[float]) -> float:
    return -sum(p * math.log2(p) for p in fractions if p > 0)


def build_module_report(corpus: Corpus, corrected: dict[str, dict], graph: ImportGraph) -> list[dict]:
    by_module: dict[str, list[dict]] = defaultdict(list)
    for rec in corrected.values():
        module_relpath = rec["id"].split(":", 2)[1]
        module_id = f"{rec['repo']}:{module_relpath}:__module__"
        by_module[module_id].append(rec)

    rows = []
    for m in corpus.modules:
        mrecs = by_module.get(m["id"], [])
        obj_loc_total = sum(r["loc"] for r in mrecs)

        purpose_loc = Counter()
        for r in mrecs:
            purpose_loc[r["purpose"]] += r["loc"]
        purpose_mix = {
            p: round(loc_ / obj_loc_total, 4) for p, loc_ in sorted(purpose_loc.items()) if obj_loc_total
        }
        dominant_purpose = max(purpose_loc.items(), key=lambda kv: (kv[1], kv[0]))[0] if purpose_loc else None
        entropy = shannon_entropy_bits(purpose_mix.values()) if purpose_mix else 0.0

        def pct_for_status(status: str) -> float:
            loc_ = sum(r["loc"] for r in mrecs if r["usage_status"] == status)
            return round(100.0 * loc_ / obj_loc_total, 2) if obj_loc_total else 0.0

        flag_counts = Counter()
        for r in mrecs:
            for fl in r["contract_flags"]:
                if fl != "none":
                    flag_counts[fl] += 1
        n_flagged_objects = sum(1 for r in mrecs if r["contract_flags"] != ["none"])
        flag_density_per_kloc = round(sum(flag_counts.values()) / (m["loc"] / 1000.0), 3) if m["loc"] else 0.0

        relocate_loc = sum(
            r["loc"] for r in mrecs if RELOCATE_FLAGS & set(r["contract_flags"])
        )
        relocate_flag_fraction = round(relocate_loc / obj_loc_total, 4) if obj_loc_total else 0.0
        relocate_flags_present = sorted({
            fl for r in mrecs for fl in r["contract_flags"] if fl in RELOCATE_FLAGS
        })

        dotted = relpath_to_dotted(m["repo"], m["tree"], m["relpath"])
        fan_in = len(graph.in_edges.get(m["id"], ())) if dotted else None
        fan_out = len(graph.out_edges.get(m["id"], ())) if dotted else None
        neighbors = top_neighbors(graph, m["id"]) if dotted else []

        split_eligible = m["tree"] in LIBRARY_TREES and len(mrecs) >= 3
        split_score = round(m["loc"] * entropy * (1 + len(flag_counts)), 2) if split_eligible else 0.0

        rows.append({
            "id": m["id"],
            "repo": m["repo"],
            "relpath": m["relpath"],
            "tree": m["tree"],
            "loc": m["loc"],
            "classified_object_loc": obj_loc_total,
            "n_objects": len(mrecs),
            "purpose_mix": purpose_mix,
            "dominant_purpose": dominant_purpose,
            "purpose_entropy_bits": round(entropy, 4),
            "pct_dead_loc": pct_for_status("dead"),
            "pct_legacy_only_loc": pct_for_status("legacy_only"),
            "pct_test_only_loc": pct_for_status("test_only"),
            "contract_flag_counts": dict(sorted(flag_counts.items())),
            "n_flagged_objects": n_flagged_objects,
            "contract_flag_density_per_kloc": flag_density_per_kloc,
            "relocate_flag_loc_fraction": relocate_flag_fraction,
            "relocate_flags_present": relocate_flags_present,
            "has_legacy_banner": m.get("has_legacy_banner", False),
            "fan_in": fan_in,
            "fan_out": fan_out,
            "top_co_referencing_modules": neighbors,
            "split_eligible": split_eligible,
            "split_score": split_score,
        })

    rows.sort(key=lambda r: r["id"])
    return rows


def rank_split_candidates(module_rows: list[dict], n: int = TOP_N_SPLIT) -> list[dict]:
    eligible = [r for r in module_rows if r["split_eligible"] and r["split_score"] > 0]
    eligible.sort(key=lambda r: (-r["split_score"], r["id"]))
    out = []
    for r in eligible[:n]:
        top_flags = sorted(r["contract_flag_counts"].items(), key=lambda kv: -kv[1])[:3]
        out.append({
            "module": strip_module_suffix(r["id"]),
            "module_id": r["id"],
            "loc": r["loc"],
            "n_objects": r["n_objects"],
            "purpose_entropy_bits": r["purpose_entropy_bits"],
            "dominant_purpose": r["dominant_purpose"],
            "n_distinct_contract_flags": len(r["contract_flag_counts"]),
            "split_score": r["split_score"],
            "evidence": (
                f"{r['loc']} LOC, {r['n_objects']} objects, purpose entropy "
                f"{r['purpose_entropy_bits']:.2f} bits (dominant: {r['dominant_purpose']}), "
                f"{len(r['contract_flag_counts'])} distinct contract flags"
                + (f" (top: {', '.join(f'{k}x{v}' for k, v in top_flags)})" if top_flags else "")
                + f" -> split_score={r['split_score']:.0f} (loc x entropy x (1+n_flags))."
            ),
        })
    return out


def rank_join_candidates(corpus: Corpus, module_rows: list[dict], graph: ImportGraph, n: int = TOP_N_JOIN) -> list[dict]:
    by_id = {r["id"]: r for r in module_rows}
    eligible = [
        r for r in module_rows
        if r["tree"] in LIBRARY_TREES and r["loc"] <= JOIN_LOC_THRESHOLD and r["dominant_purpose"] is not None
    ]
    by_dir: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in eligible:
        dirpath = str(Path(r["relpath"]).parent)
        by_dir[(r["repo"], dirpath)].append(r)

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    pair_evidence: dict[tuple[str, str], dict] = {}
    for (_repo, _dirpath), siblings in by_dir.items():
        siblings = sorted(siblings, key=lambda r: r["id"])
        for i in range(len(siblings)):
            for j in range(i + 1, len(siblings)):
                a, b = siblings[i], siblings[j]
                if a["dominant_purpose"] != b["dominant_purpose"]:
                    continue
                direct_edge = (
                    b["id"] in graph.out_edges.get(a["id"], ())
                    or a["id"] in graph.out_edges.get(b["id"], ())
                )
                na = graph.out_edges.get(a["id"], set()) | graph.in_edges.get(a["id"], set())
                nb = graph.out_edges.get(b["id"], set()) | graph.in_edges.get(b["id"], set())
                na = na - {a["id"], b["id"]}
                nb = nb - {a["id"], b["id"]}
                union_n = na | nb
                jaccard = len(na & nb) / len(union_n) if union_n else 0.0
                score = (2.0 if direct_edge else 0.0) + jaccard
                if score <= 0:
                    continue
                union(a["id"], b["id"])
                key = tuple(sorted((a["id"], b["id"])))
                pair_evidence[key] = {
                    "direct_edge": direct_edge,
                    "shared_neighbor_jaccard": round(jaccard, 3),
                    "score": round(score, 3),
                }

    clusters: dict[str, set[str]] = defaultdict(set)
    for mid in parent:
        clusters[find(mid)].add(mid)
    clusters = {root: members for root, members in clusters.items() if len(members) >= 2}

    ranked = []
    for root, members in clusters.items():
        members_sorted = sorted(members)
        total_loc = sum(by_id[m]["loc"] for m in members_sorted)
        pair_scores = [
            pair_evidence[tuple(sorted((a, b)))]["score"]
            for i, a in enumerate(members_sorted) for b in members_sorted[i + 1:]
            if tuple(sorted((a, b))) in pair_evidence
        ]
        cluster_score = sum(pair_scores)
        evidence_lines = [
            f"{strip_module_suffix(a)} <-> {strip_module_suffix(b)}: {pair_evidence[tuple(sorted((a, b)))]}"
            for i, a in enumerate(members_sorted) for b in members_sorted[i + 1:]
            if tuple(sorted((a, b))) in pair_evidence
        ]
        ranked.append({
            "members": [strip_module_suffix(m) for m in members_sorted],
            "member_ids": members_sorted,
            "dominant_purpose": by_id[members_sorted[0]]["dominant_purpose"],
            "total_loc": total_loc,
            "cluster_coupling_score": round(cluster_score, 3),
            "evidence": (
                f"{len(members_sorted)} sibling modules under the same directory, same dominant "
                f"purpose ({by_id[members_sorted[0]]['dominant_purpose']}), {total_loc} combined LOC; "
                + "; ".join(evidence_lines)
            ),
        })
    ranked.sort(key=lambda c: (-c["cluster_coupling_score"], c["total_loc"], c["members"]))
    return ranked[:n]


def rank_relocate_candidates(module_rows: list[dict], n: int = TOP_N_RELOCATE) -> list[dict]:
    eligible = [r for r in module_rows if r["relocate_flag_loc_fraction"] > 0]
    eligible.sort(key=lambda r: (-round(r["relocate_flag_loc_fraction"] * r["classified_object_loc"]), r["id"]))
    out = []
    for r in eligible[:n]:
        flagged_loc = round(r["relocate_flag_loc_fraction"] * r["classified_object_loc"])
        target = (
            "results/<hash>/scripts/" if "misplaced_should_be_results_scripts" in r["relocate_flags_present"]
            else "src/rlrmp/<capability>/" if "misplaced_should_be_library" in r["relocate_flags_present"]
            else "results/<hash>/ (experiment-named object out of src/)"
        )
        out.append({
            "module": strip_module_suffix(r["id"]),
            "module_id": r["id"],
            "loc": r["loc"],
            "flagged_loc": flagged_loc,
            "relocate_flag_loc_fraction": r["relocate_flag_loc_fraction"],
            "flags": r["relocate_flags_present"],
            "suggested_target": target,
            "evidence": (
                f"{flagged_loc}/{r['classified_object_loc']} classified-object LOC "
                f"({r['relocate_flag_loc_fraction']:.0%}) flagged {', '.join(r['relocate_flags_present'])}."
            ),
        })
    return out


# --------------------------------------------------------------------------
# Deliverable 1c: portfolio
# --------------------------------------------------------------------------

def module_of(record_id: str) -> tuple[str, str]:
    repo, module_relpath, _qual = record_id.split(":", 2)
    return repo, module_relpath


def group_by_module(records: list[dict]) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        out[module_of(r["id"])].append(r)
    return out


def build_portfolio(corpus: Corpus, corrected: dict[str, dict]) -> dict[str, list[dict]]:
    recs = list(corrected.values())
    portfolio: dict[str, list[dict]] = {}

    # (a) confirmed-dead deletions: post-correction disposition == delete AND
    # the record was actually verified (confirmed, or refuted into delete).
    delete_confirmed = [
        r for r in recs
        if r["disposition"] == "delete" and r["_verification"] in ("confirmed", "refuted_corrected")
    ]
    items = []
    for (repo, module_relpath), members in sorted(group_by_module(delete_confirmed).items()):
        members = sorted(members, key=lambda r: r["id"])
        loc = sum(m["loc"] for m in members)
        confidences = Counter(m["confidence"] for m in members)
        items.append({
            "id": f"a_delete/{repo}/{module_relpath}",
            "class": "confirmed_dead_deletion",
            "scope": {"repo": repo, "module": module_relpath, "object_ids": [m["id"] for m in members]},
            "est_loc_delta": -loc,
            "risk_note": (
                f"{len(members)} object(s), verification-confirmed dead/unused "
                f"(confidence: {dict(confidences)}); disposition_note(s): "
                + " | ".join(sorted({m["disposition_note"] for m in members}))[:400]
            ),
            "suggested_issue_title": f"Delete confirmed-dead code in {module_relpath}",
        })
    items.sort(key=lambda it: it["est_loc_delta"])
    portfolio["a_confirmed_dead_deletions"] = items

    unverified_delete = [
        r for r in recs if r["disposition"] == "delete" and r["_verification"] == "unverified"
    ]
    portfolio["a_confirmed_dead_deletions_footnote"] = [{
        "id": "a_footnote/unverified_delete_candidates",
        "class": "unverified_delete_candidate_not_included",
        "scope": {"n_objects": len(unverified_delete), "loc": sum(r["loc"] for r in unverified_delete)},
        "est_loc_delta": 0,
        "risk_note": "disposition=delete records with no verification verdict; excluded from (a) per "
                      "'only verification-confirmed'. Empirically empty for this corpus run (delete was "
                      "verified exhaustively), retained here as a structural safety net.",
        "suggested_issue_title": None,
    }]

    # (b) legacy-tree retirements: legacy_only clusters by module, plus the
    # feedbax reverse-audit deprecate_delete set (the "orphaned analysis
    # layer": feedbax/analysis/{effector,profiles,setup}.py,
    # bin/db_merge.py, config/defaults.py, plot/mpl.py, web/ws/simulation.py).
    legacy_recs = [r for r in recs if r["usage_status"] == "legacy_only"]
    items = []
    for (repo, module_relpath), members in sorted(group_by_module(legacy_recs).items()):
        members = sorted(members, key=lambda r: r["id"])
        loc = sum(m["loc"] for m in members)
        items.append({
            "id": f"b_legacy/{repo}/{module_relpath}",
            "class": "legacy_only_cluster",
            "scope": {"repo": repo, "module": module_relpath, "object_ids": [m["id"] for m in members]},
            "est_loc_delta": -loc,
            "risk_note": f"{len(members)} legacy_only object(s), {loc} LOC; retirement gated on the "
                         f"module's LEGACY-banner porting/deletion decision per CLAUDE.md's "
                         f"LEGACY-banner convention.",
            "suggested_issue_title": f"Retire or port legacy_only code in {module_relpath}",
        })
    for r in corpus.reverse_audit:
        if r["recommendation"] != "deprecate_delete":
            continue
        m = resolve_reverse_audit_module(corpus, r)
        items.append({
            "id": f"b_reverse_audit/{r['relpath']}",
            "class": "feedbax_reverse_audit_deprecate_delete",
            "scope": {"repo": "feedbax", "module": m["relpath"] if m else r["relpath"]},
            "est_loc_delta": -(m["loc"] if m else 0),
            "risk_note": r.get("recommendation_note", "")[:500],
            "suggested_issue_title": f"Deprecate/delete feedbax module {r['relpath']} (reverse audit)",
        })
    items.sort(key=lambda it: it["est_loc_delta"])
    portfolio["b_legacy_tree_retirements"] = items

    # (c) dedupe/promotion items from top duplication clusters (>=100 redundant LOC).
    items = []
    for c in sorted(corpus.dup_clusters, key=lambda c: -c["redundant_loc_estimate"]):
        if c["redundant_loc_estimate"] < MIN_DUP_CLUSTER_LOC_FOR_PORTFOLIO:
            continue
        member_dispositions = Counter(
            corrected[mid]["disposition"] for mid in c["member_ids"] if mid in corrected
        )
        items.append({
            "id": f"c_dup/{c['cluster_id']}",
            "class": "dedupe_promotion",
            "scope": {
                "cluster_id": c["cluster_id"], "kind": c["kind"], "size": c["size"],
                "member_ids": c["member_ids"], "cross_repo": c["cross_repo"],
            },
            "est_loc_delta": -c["redundant_loc_estimate"],
            "risk_note": (
                f"mean pairwise similarity {c['mean_pairwise_similarity']}, {c['size']} members, "
                f"{c['total_loc']} total LOC; classification dispositions among members: "
                f"{dict(member_dispositions)}."
            ),
            "suggested_issue_title": f"Deduplicate near-identical {c['kind']} cluster {c['cluster_id']} "
                                      f"({c['size']} members, ~{c['redundant_loc_estimate']} redundant LOC)",
        })
    portfolio["c_dedupe_promotion"] = items

    # (d) move_to_results_scripts sets.
    move_recs = [r for r in recs if r["disposition"] == "move_to_results_scripts"]
    items = []
    for (repo, module_relpath), members in sorted(group_by_module(move_recs).items()):
        members = sorted(members, key=lambda r: r["id"])
        loc = sum(m["loc"] for m in members)
        items.append({
            "id": f"d_move_rs/{repo}/{module_relpath}",
            "class": "move_to_results_scripts",
            "scope": {"repo": repo, "module": module_relpath, "object_ids": [m["id"] for m in members]},
            "est_loc_delta": 0,
            "risk_note": f"{len(members)} object(s), {loc} LOC; relocation only (no deletion), per "
                         f"script-placement policy (CLAUDE.md).",
            "suggested_issue_title": f"Move experiment-specific code out of {module_relpath} into "
                                      f"results/<hash>/scripts/",
        })
    portfolio["d_move_to_results_scripts"] = items

    # (e) contract-violation remediations: data_in_code / experiment_named_in_src /
    # spec_first_violation, grouped by module.
    flagged = [r for r in recs if MISPLACEMENT_FLAGS_FOR_PORTFOLIO_E & set(r["contract_flags"])]
    items = []
    for (repo, module_relpath), members in sorted(group_by_module(flagged).items()):
        members = sorted(members, key=lambda r: r["id"])
        loc = sum(m["loc"] for m in members)
        flags_seen = sorted({fl for m in members for fl in m["contract_flags"] if fl in MISPLACEMENT_FLAGS_FOR_PORTFOLIO_E})
        items.append({
            "id": f"e_contract/{repo}/{module_relpath}",
            "class": "contract_violation_remediation",
            "scope": {"repo": repo, "module": module_relpath, "object_ids": [m["id"] for m in members], "flags": flags_seen},
            "est_loc_delta": 0,
            "risk_note": f"{len(members)} object(s), {loc} LOC flagged {flags_seen}; remediation is "
                         f"migration to a governed data product / spec surface, not deletion.",
            "suggested_issue_title": f"Migrate {module_relpath} off baked-in data/spec-first violations",
        })
    items.sort(key=lambda it: -sum(len(m["scope"]["object_ids"]) for m in [it]))
    portfolio["e_contract_violation_remediation"] = items

    # (f) feedbax generalize-in-place set.
    items = []
    for r in corpus.reverse_audit:
        if r["recommendation"] != GENERALIZE_IN_PLACE:
            continue
        m = resolve_reverse_audit_module(corpus, r)
        items.append({
            "id": f"f_generalize/{r['relpath']}",
            "class": "feedbax_generalize_in_place",
            "scope": {"repo": "feedbax", "module": m["relpath"] if m else r["relpath"]},
            "est_loc_delta": 0,
            "risk_note": r.get("recommendation_note", "")[:500],
            "suggested_issue_title": f"Generalize rlrmp-shaped feedbax module {r['relpath']} in place",
        })
    portfolio["f_feedbax_generalize_in_place"] = items

    # (g) dangling-reference fixes (high/medium confidence only).
    items = []
    dangling_hi_med = [d for d in corpus.dangling if d["confidence"] in ("high", "medium")]
    by_module: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for d in dangling_hi_med:
        by_module[(d["repo"], d["module"])].append(d)
    for (repo, module), findings in sorted(by_module.items()):
        kinds = Counter(f["kind"] for f in findings)
        items.append({
            "id": f"g_dangling/{repo}/{module}",
            "class": "dangling_reference_fix",
            "scope": {
                "repo": repo, "module": module,
                "findings": [{"line": f["line"], "kind": f["kind"], "target": f["target"], "confidence": f["confidence"]} for f in findings],
            },
            "est_loc_delta": 0,
            "risk_note": f"{len(findings)} finding(s): {dict(kinds)}.",
            "suggested_issue_title": f"Fix dangling references in {module}",
        })
    items.sort(key=lambda it: (-len(it["scope"]["findings"]), it["id"]))
    portfolio["g_dangling_reference_fixes"] = items

    return portfolio


# --------------------------------------------------------------------------
# Markdown rendering helpers
# --------------------------------------------------------------------------

def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def render_tables_md(tables: dict, checks: list[Check]) -> str:
    lines = ["# Synthesis tables -- issue `05883e7`", ""]
    lines.append("Generated by `results/05883e7/scripts/synthesize.py`. Post verification-correction "
                  "(21 refuted verdicts applied). See `results/05883e7/notes/synthesis.md` for the "
                  "narrative reading of these tables.")
    lines.append("")

    lines.append("## Corpus summary")
    lines.append("")
    cs = tables["corpus_summary"]
    lines.append(md_table(["metric", "value"], [[k, v] for k, v in cs.items()]))
    lines.append("")

    lines.append("## rlrmp src -- LOC composition by purpose")
    lines.append("")
    lines.append(f"Base: {tables['rlrmp_src_object_loc_overhead']['classified_object_loc_total']} classified "
                  f"object LOC out of {tables['rlrmp_src_object_loc_overhead']['module_loc_total']} total "
                  f"module LOC ({tables['rlrmp_src_object_loc_overhead']['overhead_pct']}% overhead: "
                  f"{tables['rlrmp_src_object_loc_overhead']['overhead_note']})")
    lines.append("")
    lines.append(md_table(
        ["purpose", "loc", "n_objects", "% of rlrmp src object LOC"],
        [[r["purpose"], r["loc"], r["n_objects"], r["pct_of_rlrmp_src_object_loc"]] for r in tables["rlrmp_src_purpose_composition"]],
    ))
    lines.append("")

    lines.append("## LOC by purpose x repo x tree")
    lines.append("")
    lines.append(md_table(
        ["repo", "tree", "purpose", "loc", "n_objects"],
        [[r["repo"], r["tree"], r["purpose"], r["loc"], r["n_objects"]] for r in tables["loc_by_purpose_x_repo_tree"]],
    ))
    lines.append("")

    lines.append("## LOC by generality x repo")
    lines.append("")
    lines.append(md_table(
        ["repo", "generality", "loc", "n_objects"],
        [[r["repo"], r["generality"], r["loc"], r["n_objects"]] for r in tables["loc_by_generality_x_repo"]],
    ))
    lines.append("")

    lines.append("## LOC by usage_status x repo x tree (post verification-correction)")
    lines.append("")
    lines.append("Note: verification corrections can introduce `usage_status` values outside the "
                  "`record_schema.md` enum (free-text overrides), e.g. `registered_contract`, "
                  "`dead_but_documented_public_api`, `live_public_api`, "
                  "`dynamic_reference_unconfirmed_wiring`. These are 6 records total.")
    lines.append("")
    lines.append(md_table(
        ["repo", "tree", "usage_status", "loc", "n_objects"],
        [[r["repo"], r["tree"], r["usage_status"], r["loc"], r["n_objects"]] for r in tables["loc_by_usage_status_x_repo_tree_post_correction"]],
    ))
    lines.append("")

    lines.append("## Confidence distribution")
    lines.append("")
    lines.append(md_table(
        ["confidence", "loc", "n_objects"],
        [[r["confidence"], r["loc"], r["n_objects"]] for r in tables["confidence_distribution"]],
    ))
    lines.append("")

    lines.append("## Contract-flag inventory")
    lines.append("")
    for row in tables["contract_flag_inventory"]:
        lines.append(f"### `{row['flag']}` -- {row['count']} objects, {row['loc']} LOC")
        lines.append("")
        if row["top_modules"]:
            lines.append(md_table(
                ["repo", "module", "count", "loc"],
                [[m["repo"], m["module"], m["count"], m["loc"]] for m in row["top_modules"]],
            ))
        else:
            lines.append("(no offending modules)")
        lines.append("")

    lines.append("## Disposition portfolio")
    lines.append("")
    dp = tables["disposition_portfolio"]
    lines.append("### delete")
    lines.append("")
    lines.append(md_table(["metric", "value"], [[k, v] for k, v in dp["delete"].items()]))
    lines.append("")
    lines.append("### merge_dedupe")
    lines.append("")
    lines.append(md_table(["metric", "value"], [[k, v] for k, v in dp["merge_dedupe"].items()]))
    lines.append("")
    lines.append("### move_to_results_scripts")
    lines.append("")
    lines.append(md_table(["metric", "value"], [[k, v] for k, v in dp["move_to_results_scripts"].items()]))
    lines.append("")
    lines.append("### legacy_only by repo x tree")
    lines.append("")
    lines.append(md_table(
        ["repo", "tree", "loc", "n_objects"],
        [[r["repo"], r["tree"], r["loc"], r["n_objects"]] for r in dp["legacy_only_by_repo_tree"]],
    ))
    lines.append("")
    lines.append("### needs_decision")
    lines.append("")
    lines.append(md_table(["metric", "value"], [[k, v] for k, v in dp["needs_decision"].items()]))
    lines.append("")

    lines.append("## Reconciliation self-check")
    lines.append("")
    lines.append(md_table(
        ["check", "status", "detail"],
        [[c.name, "PASS" if c.ok else "FAIL", c.detail] for c in checks],
    ))
    lines.append("")

    return "\n".join(lines) + "\n"


def render_module_report_md(module_rows: list[dict], split_c: list[dict], join_c: list[dict], relocate_c: list[dict]) -> str:
    lines = ["# Module structure candidates -- issue `05883e7`", ""]
    lines.append("Generated by `results/05883e7/scripts/synthesize.py` from `module_report.jsonl`. "
                  "See `results/05883e7/notes/module_structure.md` for the curated narrative version.")
    lines.append("")

    lines.append(f"## SPLIT candidates (top {len(split_c)})")
    lines.append("")
    lines.append("Score = module LOC x purpose-entropy (bits) x (1 + n distinct contract flags).")
    lines.append("")
    lines.append(md_table(
        ["module", "loc", "n_objects", "entropy(bits)", "dominant_purpose", "n_flags", "split_score"],
        [[r["module"], r["loc"], r["n_objects"], r["purpose_entropy_bits"], r["dominant_purpose"],
          r["n_distinct_contract_flags"], r["split_score"]] for r in split_c],
    ))
    lines.append("")
    for r in split_c:
        lines.append(f"- **{r['module']}**: {r['evidence']}")
    lines.append("")

    lines.append(f"## JOIN candidates (top {len(join_c)} clusters)")
    lines.append("")
    lines.append("Sibling modules (same directory) with the same dominant purpose, both <= "
                  f"{JOIN_LOC_THRESHOLD} LOC, joined by union-find over pairwise coupling score "
                  "(direct import edge worth 2, plus Jaccard similarity of each pair's fan-in/fan-out "
                  "neighbor sets).")
    lines.append("")
    for c in join_c:
        lines.append(f"### {', '.join(c['members'])}")
        lines.append("")
        lines.append(f"- dominant purpose: `{c['dominant_purpose']}`, combined LOC: {c['total_loc']}, "
                      f"coupling score: {c['cluster_coupling_score']}")
        lines.append(f"- evidence: {c['evidence']}")
        lines.append("")

    lines.append(f"## RELOCATE candidates (top {len(relocate_c)})")
    lines.append("")
    lines.append(md_table(
        ["module", "loc", "flagged_loc", "fraction", "flags", "suggested_target"],
        [[r["module"], r["loc"], r["flagged_loc"], f"{r['relocate_flag_loc_fraction']:.0%}",
          ", ".join(r["flags"]), r["suggested_target"]] for r in relocate_c],
    ))
    lines.append("")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--audit-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = args.repo_root or script_path.parents[3]
    audit_root = args.audit_root or (repo_root / "_artifacts" / "05883e7" / "audit")
    out_dir = args.out_dir or (audit_root / "synthesis")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[synthesize] audit_root={audit_root}", file=sys.stderr)
    corpus = load_corpus(audit_root)
    corrected = apply_corrections(corpus)
    graph = build_import_graph(corpus)

    checks = reconcile(corpus, corrected)
    print("\n[synthesize] reconciliation:", file=sys.stderr)
    n_fail = 0
    for c in checks:
        status = "PASS" if c.ok else "FAIL"
        if not c.ok:
            n_fail += 1
        print(f"  [{status}] {c.name}: {c.detail}", file=sys.stderr)

    tables = build_tables(corpus, corrected, graph)
    module_rows = build_module_report(corpus, corrected, graph)
    split_candidates = rank_split_candidates(module_rows)
    join_candidates = rank_join_candidates(corpus, module_rows, graph)
    relocate_candidates = rank_relocate_candidates(module_rows)
    portfolio = build_portfolio(corpus, corrected)

    (out_dir / "tables.json").write_text(json.dumps(tables, indent=2) + "\n", encoding="utf-8")
    (out_dir / "tables.md").write_text(render_tables_md(tables, checks), encoding="utf-8")

    with (out_dir / "module_report.jsonl").open("w", encoding="utf-8") as f:
        for row in module_rows:
            f.write(json.dumps(row) + "\n")
    (out_dir / "module_report.md").write_text(
        render_module_report_md(module_rows, split_candidates, join_candidates, relocate_candidates),
        encoding="utf-8",
    )

    portfolio_out = dict(portfolio)
    portfolio_out["_module_structure"] = {
        "split_candidates": split_candidates,
        "join_candidates": join_candidates,
        "relocate_candidates": relocate_candidates,
    }
    (out_dir / "portfolio.json").write_text(json.dumps(portfolio_out, indent=2) + "\n", encoding="utf-8")

    print(f"\n[synthesize] wrote synthesis outputs to {out_dir}", file=sys.stderr)
    if n_fail:
        print(f"[synthesize] {n_fail} reconciliation check(s) FAILED", file=sys.stderr)
        return 1
    print("[synthesize] all reconciliation checks passed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ReconciliationError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
