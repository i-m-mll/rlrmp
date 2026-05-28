"""Write the d01c35a linear equivalence certificate artifacts."""

from __future__ import annotations

from rlrmp.analysis.linear_equivalence_certificate import ISSUE_ID, write_outputs


def main() -> None:
    manifest = write_outputs(issue_id=ISSUE_ID)
    print(f"Wrote {manifest['tracked_note']}")
    print(f"Wrote {manifest['tracked_manifest']}")
    print(f"Wrote {manifest['artifact_npz']}")


if __name__ == "__main__":
    main()
