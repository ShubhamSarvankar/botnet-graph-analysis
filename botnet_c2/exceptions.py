"""Custom exception hierarchy for botnet_c2.

All exceptions raised by this package inherit from BotnetC2Error so callers
can catch the entire family with a single except clause if needed.

Hierarchy:
    BotnetC2Error
    ├── DataError
    │   ├── DownloadError
    │   └── SchemaError
    ├── GraphError
    └── ModelError
"""


class BotnetC2Error(Exception):
    """Base exception for all botnet_c2 errors."""


# ── Data layer ────────────────────────────────────────────────────────────────


class DataError(BotnetC2Error):
    """Raised for failures in data download, loading, or validation."""


class DownloadError(DataError):
    """Raised when a capture file cannot be downloaded.

    Attributes:
        scenario_id: The scenario key that failed (e.g. "neris_42").
        url: The URL that was attempted.
    """

    def __init__(self, scenario_id: str, url: str, reason: str) -> None:
        self.scenario_id = scenario_id
        self.url = url
        self.reason = reason
        super().__init__(
            f"Failed to download scenario '{scenario_id}' from {url}: {reason}"
        )


class SchemaError(DataError):
    """Raised when a loaded DataFrame is missing required columns or has wrong dtypes.

    Attributes:
        missing_columns: Columns present in the schema spec but absent in the file.
    """

    def __init__(self, path: str, missing_columns: list[str]) -> None:
        self.path = path
        self.missing_columns = missing_columns
        super().__init__(
            f"Schema validation failed for '{path}'. Missing columns: {missing_columns}"
        )


# ── Graph layer ───────────────────────────────────────────────────────────────


class GraphError(BotnetC2Error):
    """Raised for failures in graph construction or metric computation."""


# ── Model layer ───────────────────────────────────────────────────────────────


class ModelError(BotnetC2Error):
    """Raised for failures in model training, evaluation, or serialization."""
