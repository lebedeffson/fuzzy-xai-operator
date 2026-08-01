"""Read-only adapters for independently produced ML/XAI artifacts."""

from .base import ExternalPipelineAdapter, ExternalPipelineArtifacts
from .manifest import ManifestExternalPipelineAdapter

__all__ = [
    "ExternalPipelineAdapter",
    "ExternalPipelineArtifacts",
    "ManifestExternalPipelineAdapter",
]
