Artifact manifest normalization fixture for Mandible and Feedbax.

This directory is a checked-in fixture, not a training result. It documents how
RLRMP's native dual records map into portable provider/artifact references:
tracked run specs stay under `results/<issue>/runs/*.json`, bulk outputs stay
under ignored `_artifacts/<issue>/runs/<variant>/`, and Feedbax manifests can
refer to both without relocating either source of truth. Stale absolute
worktree paths are represented only as `metadata.normalized_from`; all loadable
`uri` fields are repo-relative. The sample manifest is intentionally compatible
with the current public Feedbax `TrainingRunManifest`/`ArtifactRef` contract and
does not define a permanent Mandible or Feedbax schema.
