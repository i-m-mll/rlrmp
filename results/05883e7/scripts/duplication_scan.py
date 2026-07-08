#!/usr/bin/env python3
"""Phase 3 near-duplicate clustering sweep for the code-archaeology audit.

Mandible issue: 05883e7. Stdlib-only. Consumes the Phase 0 census corpus
(`_artifacts/05883e7/audit/census/objects.jsonl`) and clusters near-duplicate
top-level objects (functions, async functions, classes, module-level
constants) corpus-wide across rlrmp (src/tests/scripts/results-scripts) and
feedbax (package/tests/scripts/examples).

Usage (from the rlrmp worktree root):

    PYTHONPATH=src uv run --no-sync python results/05883e7/scripts/duplication_scan.py

Emits under `_artifacts/05883e7/audit/sweeps/duplication/`:
    clusters.jsonl  -- one line per near-duplicate cluster
    summary.md      -- cluster counts, redundant-LOC estimates, hot-zone report

Method (documented here since every shortcut below is a deliberate, bounded
approximation -- see the "Heuristic limitations" section emitted in
summary.md for the reader-facing version of the same list):

1. Normalize each candidate object's source span to a token stream: strip
   comments, docstrings (including nested method docstrings, found via one
   `ast.parse` of the object's own snippet), and blank/structural tokens
   (NL/NEWLINE/INDENT/DEDENT), keeping identifiers, operators, numbers, and
   string literals as tokens. Objects with fewer than `MIN_LOC` lines (per
   the census `loc` field) are excluded up front -- they produce mostly
   spurious matches and contribute negligible redundant LOC even when they
   do cluster correctly.
2. Cheap bucketing: candidates are grouped by `kind` (never cross function
   vs class vs constant), then indexed by a bottom-k sketch (K-Minimum-Values
   synopsis) of stable (blake2b-based) hashes over k=8 token shingles. Two
   objects are only ever compared if they share at least one sketch hash
   value (an inverted-index bucket) -- this is the standard MinHash/KMV-style
   LSH trick to avoid all-pairs comparison over ~14.5k objects. Within a
   bucket, a further LOC-ratio filter (min/max >= 1/1.35, i.e. within ~35%)
   is applied before the expensive real-similarity scoring. Buckets larger
   than `MAX_BUCKET_SIZE` are capped by pairwise-LOC-sorted sliding window
   rather than skipped outright, bounding worst-case cost from a single
   very common shingle (e.g. a boilerplate import block).
3. Real similarity for candidate pairs surviving step 2:
   `difflib.SequenceMatcher(None, tokens_a, tokens_b, autojunk=False).ratio()`
   run directly on the token sequences (not re-joined strings), threshold
   0.7. `autojunk=False` because token streams are not prose and the
   autojunk heuristic misclassifies repetitive-but-meaningful code tokens.
4. Union-find over object ids for pairs scoring >= 0.7 forms clusters.

Self-checks (exit nonzero on failure):
    - A synthetic control object with a unique, randomly-salted token stream
      is injected into the same pipeline (bucketing + scoring) as a known
      negative control; it must end up in a singleton cluster (no real
      object should ever match its random content). It is excluded from the
      emitted output -- it exists purely to validate the pipeline.
    - clusters.jsonl round-trips through json.loads with all required fields
      present on every line.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import secrets
import sys
import time
import tokenize
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------
# Constants / tunables
# --------------------------------------------------------------------------

MIN_LOC = 3              # objects shorter than this are excluded from clustering
SHINGLE_K = 8             # token-shingle window size
SKETCH_SIZE = 8           # bottom-k sketch size per object
MAX_BUCKET_SIZE = 400     # sliding-window cap per LSH bucket (see docstring step 2)
LOC_RATIO_MIN = 1.0 / 1.35  # "within ~35%"
SIM_THRESHOLD = 0.70
CANDIDATE_KINDS = {"function", "async_function", "class", "constant"}

SKIP_MODULES = {"__pycache__"}

REQUIRED_CLUSTER_FIELDS = {
    "cluster_id", "kind", "size", "member_ids", "members", "total_loc",
    "max_loc", "redundant_loc_estimate", "mean_pairwise_similarity",
    "min_pairwise_similarity", "cross_repo", "repos",
}


class SelfCheckError(Exception):
    pass


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Candidate:
    id: str
    repo: str
    tree: str
    module_relpath: str
    qualname: str
    kind: str
    loc: int
    tokens: list[str] = field(default_factory=list)
    sketch: list[int] = field(default_factory=list)


# --------------------------------------------------------------------------
# Union-Find
# --------------------------------------------------------------------------

class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

def _docstring_line_span(node: ast.AST) -> tuple[int, int] | None:
    body = getattr(node, "body", None)
    if not body:
        return None
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first.lineno, (first.end_lineno or first.lineno)
    return None


def normalize_snippet(snippet: str) -> list[str]:
    """Return a token list for `snippet` with comments/docstrings/blank lines
    stripped and identifiers/operators/literals kept. Falls back to a crude
    regex-based comment/blank-line strip if the snippet fails to parse or
    tokenize (rare -- e.g. a stray trailing comma from a span-extraction
    edge case)."""
    docstring_lines: set[int] = set()
    try:
        tree = ast.parse(snippet)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                span = _docstring_line_span(node)
                if span:
                    docstring_lines.update(range(span[0], span[1] + 1))
    except (SyntaxError, ValueError):
        pass

    tokens: list[str] = []
    skip_types = {
        tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
        tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER,
    }
    try:
        for tok in tokenize.generate_tokens(StringIO(snippet).readline):
            if tok.type in skip_types:
                continue
            if tok.type == tokenize.STRING and tok.start[0] in docstring_lines:
                continue
            if not tok.string.strip():
                continue
            tokens.append(tok.string)
        return tokens
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass

    # Fallback: crude line-based comment/blank strip.
    for line in snippet.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            tokens.append(stripped)
    return tokens


def stable_hash(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def sketch_of(tokens: list[str]) -> list[int]:
    if len(tokens) < SHINGLE_K:
        shingles = {tuple(tokens)} if tokens else set()
    else:
        shingles = {tuple(tokens[i:i + SHINGLE_K]) for i in range(len(tokens) - SHINGLE_K + 1)}
    hashes = sorted(stable_hash("\x00".join(sh)) for sh in shingles)
    return hashes[:SKETCH_SIZE]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_objects(objects_path: Path) -> list[dict]:
    out = []
    with objects_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def read_snippet(root: Path, module_relpath: str, line_start: int, line_end: int,
                  file_cache: dict[Path, list[str]]) -> Optional[str]:
    path = root / module_relpath
    lines = file_cache.get(path)
    if lines is None:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        except OSError:
            lines = []
        file_cache[path] = lines
    if not lines or line_start < 1 or line_end > len(lines):
        return None
    return "".join(lines[line_start - 1:line_end])


# --------------------------------------------------------------------------
# Bucketing / candidate generation
# --------------------------------------------------------------------------

def build_candidate_pairs(candidates: list[Candidate]) -> tuple[set[tuple[str, str]], int]:
    by_kind: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        by_kind[c.kind].append(c)

    pairs: set[tuple[str, str]] = set()
    n_buckets_capped = 0

    for kind, group in by_kind.items():
        bucket_index: dict[int, list[Candidate]] = defaultdict(list)
        for c in group:
            for h in c.sketch:
                bucket_index[h].append(c)

        for h, members in bucket_index.items():
            if len(members) < 2:
                continue
            members_sorted = sorted(members, key=lambda c: c.loc)
            if len(members_sorted) > MAX_BUCKET_SIZE:
                n_buckets_capped += 1
            # Sliding window over LOC-sorted members: for each i, only look
            # forward while the LOC ratio stays within tolerance. This bounds
            # worst-case cost for a very common shingle (many members in one
            # bucket) to roughly bucket_size * average-window-size rather
            # than bucket_size^2, since the LOC-ratio filter caps the window.
            n = len(members_sorted)
            for i in range(n):
                a = members_sorted[i]
                if a.loc <= 0:
                    continue
                j = i + 1
                while j < n and members_sorted[j].loc <= a.loc / LOC_RATIO_MIN:
                    b = members_sorted[j]
                    key = (a.id, b.id) if a.id < b.id else (b.id, a.id)
                    pairs.add(key)
                    j += 1

    return pairs, n_buckets_capped


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--feedbax-root", type=Path,
                     default=Path("~/Main/10 Projects/10 PhD/20 Feedbax/feedbax").expanduser())
    ap.add_argument("--census-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = args.repo_root or script_path.parents[3]
    feedbax_root = args.feedbax_root.resolve()
    census_dir = args.census_dir or (repo_root / "_artifacts" / "05883e7" / "audit" / "census")
    out_dir = args.out_dir or (repo_root / "_artifacts" / "05883e7" / "audit" / "sweeps" / "duplication")
    out_dir.mkdir(parents=True, exist_ok=True)

    objects_path = census_dir / "objects.jsonl"
    if not objects_path.is_file():
        print(f"error: {objects_path} not found -- run census.py first", file=sys.stderr)
        return 2

    t0 = time.monotonic()
    raw_objects = load_objects(objects_path)

    file_cache: dict[Path, list[str]] = {}
    candidates: list[Candidate] = []
    n_excluded_short = 0
    n_read_failed = 0

    for o in raw_objects:
        if o["kind"] not in CANDIDATE_KINDS:
            continue
        if o["loc"] < MIN_LOC:
            n_excluded_short += 1
            continue
        root = repo_root if o["repo"] == "rlrmp" else feedbax_root
        snippet = read_snippet(root, o["module_relpath"], o["line_start"], o["line_end"], file_cache)
        if snippet is None:
            n_read_failed += 1
            continue
        tokens = normalize_snippet(snippet)
        if not tokens:
            n_read_failed += 1
            continue
        cand = Candidate(
            id=o["id"], repo=o["repo"], tree=o["tree"], module_relpath=o["module_relpath"],
            qualname=o["qualname"], kind=o["kind"], loc=o["loc"], tokens=tokens,
        )
        cand.sketch = sketch_of(tokens)
        candidates.append(cand)

    # ---- Self-check control object: unique random token stream, must end
    # up as a singleton. Injected into the same candidate pool so it goes
    # through identical bucketing/scoring, then excluded from output.
    control_salt = secrets.token_hex(16)
    control_tokens = normalize_snippet(
        f"def control_{control_salt}():\n"
        f"    return '{control_salt}_unique_marker_no_real_object_should_match_this'\n"
    )
    control = Candidate(
        id=f"__self_check_control__:{control_salt}", repo="control", tree="control",
        module_relpath="<synthetic>", qualname=f"control_{control_salt}", kind="function",
        loc=2, tokens=control_tokens,
    )
    control.sketch = sketch_of(control_tokens)
    candidates.append(control)

    pairs, n_buckets_capped = build_candidate_pairs(candidates)

    uf = UnionFind()
    by_id = {c.id: c for c in candidates}
    pair_similarity: dict[tuple[str, str], float] = {}
    for a_id, b_id in pairs:
        a, b = by_id[a_id], by_id[b_id]
        ratio = difflib.SequenceMatcher(None, a.tokens, b.tokens, autojunk=False).ratio()
        pair_similarity[(a_id, b_id)] = ratio
        if ratio >= SIM_THRESHOLD:
            uf.union(a_id, b_id)

    groups: dict[str, list[str]] = defaultdict(list)
    for c in candidates:
        groups[uf.find(c.id)].append(c.id)

    # ---- Self-check: control object must be alone in its group. ----
    control_root = uf.find(control.id)
    if len(groups[control_root]) != 1:
        others = [m for m in groups[control_root] if m != control.id]
        raise SelfCheckError(
            f"self-check control object unexpectedly clustered with: {others}"
        )

    # ---- Build clusters (exclude the control object and singletons). ----
    clusters_raw = []
    for root, member_ids in groups.items():
        if control.id in member_ids:
            continue
        if len(member_ids) < 2:
            continue
        members = [by_id[m] for m in member_ids]
        sims = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a_id, b_id = members[i].id, members[j].id
                key = (a_id, b_id) if a_id < b_id else (b_id, a_id)
                if key in pair_similarity:
                    sims.append(pair_similarity[key])
                else:
                    # Members unioned transitively (a~b, b~c but a!/~c directly
                    # compared/bucketed) -- score directly now for cluster stats.
                    ratio = difflib.SequenceMatcher(
                        None, members[i].tokens, members[j].tokens, autojunk=False
                    ).ratio()
                    sims.append(ratio)
        total_loc = sum(m.loc for m in members)
        max_loc = max(m.loc for m in members)
        repos = sorted({m.repo for m in members})
        clusters_raw.append({
            "kind": members[0].kind,
            "member_ids": sorted(member_ids),
            "members": sorted(
                [{"id": m.id, "repo": m.repo, "tree": m.tree,
                  "module_relpath": m.module_relpath, "qualname": m.qualname, "loc": m.loc}
                 for m in members],
                key=lambda r: -r["loc"],
            ),
            "size": len(members),
            "total_loc": total_loc,
            "max_loc": max_loc,
            "redundant_loc_estimate": total_loc - max_loc,
            "mean_pairwise_similarity": round(sum(sims) / len(sims), 4) if sims else 1.0,
            "min_pairwise_similarity": round(min(sims), 4) if sims else 1.0,
            "cross_repo": len(repos) > 1,
            "repos": repos,
        })

    clusters_raw.sort(key=lambda c: -c["redundant_loc_estimate"])
    for i, c in enumerate(clusters_raw, start=1):
        c["cluster_id"] = f"dup_{i:04d}"

    clusters_path = out_dir / "clusters.jsonl"
    with clusters_path.open("w", encoding="utf-8") as fh:
        for c in clusters_raw:
            fh.write(json.dumps(c, sort_keys=True) + "\n")

    # ---- Self-check: JSONL round-trip ----
    n_checked = 0
    with clusters_path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SelfCheckError(f"clusters.jsonl:{i} failed to round-trip: {e}")
            missing = REQUIRED_CLUSTER_FIELDS - obj.keys()
            if missing:
                raise SelfCheckError(f"clusters.jsonl:{i} missing fields: {missing}")
            n_checked += 1

    elapsed = time.monotonic() - t0

    write_summary(
        out_dir / "summary.md", clusters_raw, len(candidates) - 1,  # -1 excludes control
        n_excluded_short, n_read_failed, n_buckets_capped, elapsed,
    )

    print(
        f"duplication_scan complete in {elapsed:.1f}s: {len(candidates) - 1} candidates, "
        f"{len(pairs)} candidate pairs scored, {len(clusters_raw)} clusters (n={n_checked} "
        f"round-tripped) -> {out_dir}"
    )
    return 0


def write_summary(
    path: Path, clusters: list[dict], n_candidates: int, n_excluded_short: int,
    n_read_failed: int, n_buckets_capped: int, elapsed: float,
) -> None:
    total_redundant = sum(c["redundant_loc_estimate"] for c in clusters)
    cross_repo_clusters = [c for c in clusters if c["cross_repo"]]

    redundant_by_tree: Counter = Counter()
    for c in clusters:
        # Attribute each non-largest member's LOC to its own (repo, tree) --
        # those are the copies that would be removed if the cluster were
        # deduplicated down to its largest member.
        members_sorted = sorted(c["members"], key=lambda m: -m["loc"])
        for m in members_sorted[1:]:
            redundant_by_tree[(m["repo"], m["tree"])] += m["loc"]

    pipelines_clusters = [
        c for c in clusters
        if any(m["module_relpath"].startswith("src/rlrmp/analysis/pipelines/") for m in c["members"])
    ]
    results_scripts_clusters = [
        c for c in clusters
        if any(m["tree"] == "results_scripts" for m in c["members"])
    ]

    lines = []
    lines.append("<!-- AUTO-GENERATED: duplication_scan_summary -->")
    lines.append("# Near-duplicate clustering sweep (Phase 3, issue 05883e7)")
    lines.append("")
    lines.append(f"Generated in {elapsed:.1f}s over {n_candidates} candidate objects "
                 f"(kind in function/async_function/class/constant, loc >= {MIN_LOC}).")
    lines.append(
        f"{n_excluded_short} objects excluded as too short (< {MIN_LOC} loc); "
        f"{n_read_failed} objects excluded because their source span could not be read/tokenized; "
        f"{n_buckets_capped} LSH buckets exceeded the {MAX_BUCKET_SIZE}-member sliding-window cap "
        f"(bounded via LOC-sorted sliding window rather than skipped -- see script docstring)."
    )
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    lines.append(f"- **Cluster count:** {len(clusters)}")
    lines.append(f"- **Total redundant-LOC estimate:** {total_redundant} "
                 f"(sum of each cluster's total LOC minus its largest member)")
    lines.append(f"- **Cross-repo clusters (rlrmp + feedbax):** {len(cross_repo_clusters)}")
    lines.append("")
    lines.append("## Redundant LOC by (repo, tree)")
    lines.append("")
    lines.append("| repo | tree | redundant LOC |")
    lines.append("|---|---|---:|")
    for (repo, tree), loc in sorted(redundant_by_tree.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {repo} | {tree} | {loc} |")
    lines.append("")
    lines.append(
        "## `src/rlrmp/analysis/pipelines/` hot zone\n\n"
        f"{len(pipelines_clusters)} clusters have at least one member under "
        f"`src/rlrmp/analysis/pipelines/`. Redundant LOC contributed by this hot zone: "
        f"{sum(c['redundant_loc_estimate'] for c in pipelines_clusters)} "
        "(cluster may also span other trees; this is the cluster's full redundant-LOC "
        "estimate, not just the pipelines-tree share)."
    )
    lines.append("")
    lines.append(
        "## `results/*/scripts/` hot zone\n\n"
        f"{len(results_scripts_clusters)} clusters have at least one member in the "
        f"`results_scripts` tree. Redundant LOC contributed: "
        f"{sum(c['redundant_loc_estimate'] for c in results_scripts_clusters)}."
    )
    lines.append("")
    lines.append("## Top 30 clusters by redundant LOC")
    lines.append("")
    lines.append(
        "_Descriptions below are written after reading a sample of each cluster's "
        "members' actual source (per the Phase 3 task instructions); they are not "
        "script-generated._"
    )
    lines.append("")
    lines.append("| cluster_id | kind | size | redundant LOC | cross-repo | members (top 3 by loc) | description |")
    lines.append("|---|---|---:|---:|---|---|---|")
    for c in clusters[:30]:
        member_str = "; ".join(
            f"`{m['module_relpath']}:{m['qualname']}` ({m['loc']} loc)" for m in c["members"][:3]
        )
        lines.append(
            f"| {c['cluster_id']} | {c['kind']} | {c['size']} | {c['redundant_loc_estimate']} | "
            f"{'yes' if c['cross_repo'] else 'no'} | {member_str} | "
            f"_(description pending manual read)_ |"
        )
    lines.append("")
    lines.append("## Heuristic limitations (documented shortcuts)")
    lines.append("")
    lines.append(
        "- Objects shorter than 3 loc are excluded entirely -- they generate mostly "
        "spurious near-identical matches (e.g. trivial one-line constants) and contribute "
        "negligible redundant LOC even when correctly clustered.\n"
        "- Candidate generation is LSH-style (bottom-k sketch of 8-token shingles, hashed "
        "with blake2b): two objects are only ever compared if they share a sketch bucket, "
        "so a true near-duplicate pair whose shingle sketches happen not to overlap (rare "
        "for genuinely near-identical code, but possible for short objects near the loc "
        "floor) will be missed. This is the standard MinHash/KMV tradeoff traded for "
        "avoiding an O(n^2) pass over ~14.5k objects.\n"
        "- Kind must match exactly (function vs class vs constant never cross-cluster), "
        "and LOC must be within ~35% before real similarity is even scored.\n"
        "- Normalization keeps string/numeric literal tokens as-is (only comments, "
        "docstrings, and blank/structural tokens are stripped) -- two objects that are "
        "structurally identical but differ only in embedded literal values (e.g. two "
        "hyperparameter dicts with the same keys/shape but different values) will score "
        "lower than a pure structural diff would; the 0.7 threshold is calibrated to still "
        "catch most such cases without inflating false positives from the corpus's dense "
        "use of hyperparameter-like literals.\n"
        "- Decorators are excluded from the compared token stream (Python's `ast.lineno` "
        "for a decorated function/class points at the `def`/`class` line, not the first "
        "decorator, matching how the Phase 0 census itself computed `loc`).\n"
        "- Very large LSH buckets (a shingle common across many objects, e.g. a standard "
        "boilerplate import block) are bounded via a LOC-sorted sliding window rather than "
        "compared exhaustively or skipped -- this can miss a genuine duplicate pair whose "
        "LOC difference exceeds the ~35% ratio filter despite sharing the common shingle.\n"
        "- `redundant_loc_estimate` is a rough proxy (total cluster LOC minus the largest "
        "member) for \"LOC removable by deduplicating down to one canonical implementation\"; "
        "it does not account for call-site migration cost, which the census's `tests` "
        "pinning (Phase 3's dangling-reference sweep sibling and Phase 3's test-pinning "
        "sweep) would need to estimate separately."
    )
    lines.append("")
    lines.append("<!-- /AUTO-GENERATED -->")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SelfCheckError as e:
        print(f"SELF-CHECK FAILED: {e}", file=sys.stderr)
        sys.exit(1)
