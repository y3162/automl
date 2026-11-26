"""Public interfaces for versioned, read-only samples and generated data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from automl.contracts import Component


@dataclass(frozen=True)
class Sample:
    """A version-local identifier, task-defined payload and source metadata."""

    sample_id: str
    data: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DatasetVersion(ABC):
    """A view bound to a single version, including its labels and splits."""

    @abstractmethod
    def get_sample(self, sample_id: str) -> Sample:
        """Read one sample by its ID within this version."""

    @abstractmethod
    def iter_samples(self, *, split: str) -> Iterable[Sample]:
        """Stream the members of a fixed split in their dataset-defined order."""


class Dataset(Component, ABC):
    """Implement versioning and registration for a particular data source."""

    @classmethod
    def replay_files(cls, settings: dict[str, Any], root: Path) -> Mapping[str, Path]:
        """Map SHA-256 digests to portable input files to retain for replay."""
        return {}

    @classmethod
    def verify_version(cls, settings: dict[str, Any], root: Path, version_id: str) -> None:
        """Check that a recorded version can be reproduced, without writing data."""
        raise NotImplementedError(f"{cls.__name__} does not provide read-only version verification")

    @abstractmethod
    def create_version(self, request_id: str) -> str:
        """Freeze the selected inputs and splits once per creation request."""

    @abstractmethod
    def open_version(self, version_id: str) -> DatasetVersion:
        """Open the named version independently of subsequent changes."""

    @abstractmethod
    def register_generated(
        self, request_id: str, *, recipe: Any = None, content: Any = None
    ) -> str:
        """Atomically register a recipe, content, or both, returning its sample ID."""

    @abstractmethod
    def registration_status(self, request_id: str) -> Mapping[str, Any]:
        """Return progress, the successful sample ID, or the cause of failure."""

    @abstractmethod
    def get_generated(self, sample_id: str) -> Mapping[str, Any]:
        """Return recipe and content, with None for an unregistered item."""

    @abstractmethod
    def verify_generated(self, sample_id: str, regenerated: Any) -> bool:
        """Compare a regenerated value under the record's original criteria."""
