"""JSON reporting utilities for ReliabilityLab."""

import json
from pathlib import Path


def save_summary_json(
    summary,
    save_path,
    metadata=None,
):
    """Save reliability summary statistics as JSON.

    Parameters
    ----------
    summary
        Dictionary containing reliability statistics.
    save_path
        Destination JSON file.
    metadata
        Optional experiment metadata.
    """

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "metadata": metadata or {},
        "summary": summary,
    }

    with save_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    print(f"Summary saved to: {save_path}")