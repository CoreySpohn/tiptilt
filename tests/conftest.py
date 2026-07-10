"""Shared fixtures for tiptilt tests."""

from hwoutils import enable_x64, set_platform

# The deep-contrast path is x64-mandatory and the suite must be deterministic:
# pin tests to CPU x64. Builder runs opt into GPU explicitly outside the suite.
set_platform("cpu")
enable_x64()
