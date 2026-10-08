"""World backends: AmuletBackend (primary), PaperBackend (later) (DESIGN §8)."""

from terrain.backends.base import BackendError, MemoryBackend, WorldBackend

__all__ = ["BackendError", "MemoryBackend", "WorldBackend"]
