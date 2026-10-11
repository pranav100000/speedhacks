"""Small helpers shared by the synth CLIs."""

from datetime import datetime

from common.schema import CorpusRecord


def created_at_key(record: CorpusRecord) -> datetime:
    """Parse the contract's ISO-8601 created_at. Raises on any other format."""
    try:
        return datetime.fromisoformat(record.created_at)
    except ValueError as e:
        raise ValueError(
            f"{record.id}: created_at {record.created_at!r} is not ISO-8601 (contract: 2025-03-01T12:00:00Z)"
        ) from e
