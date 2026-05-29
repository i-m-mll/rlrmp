"""Write the 583d764 robust Bellman diagnostic artifacts."""

from __future__ import annotations

from rlrmp.analysis.robust_bellman import ISSUE_ID, write_outputs


def main() -> None:
    manifest = write_outputs(issue_id=ISSUE_ID)
    print(f"Wrote {manifest['tracked_note']}")
    print(f"Wrote {manifest['tracked_manifest']}")


if __name__ == "__main__":
    main()
