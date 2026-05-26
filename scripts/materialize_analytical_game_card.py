"""Write the Phase 0 C&S analytical game-card artifacts."""

from __future__ import annotations

from rlrmp.analysis.cs_game_card import ISSUE_ID, write_outputs


def main() -> None:
    manifest = write_outputs(issue_id=ISSUE_ID)
    print(f"Wrote {manifest['tracked_note']}")
    print(f"Wrote {manifest['artifact_npz']}")


if __name__ == "__main__":
    main()
