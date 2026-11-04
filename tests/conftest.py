"""Shared fixtures for tiptilt tests."""

import importlib.util
import os

from hwoutils import enable_x64, set_platform
from hypothesis import settings

# Released hwoutils versions may not include the optional budget plugin yet.
pytest_plugins = []
if importlib.util.find_spec("hwoutils.testbudget") is not None:
    pytest_plugins.append("hwoutils.testbudget.plugin")

settings.register_profile("quick", max_examples=25, deadline=None)
settings.register_profile("nightly", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))

# The deep-contrast path is x64-mandatory and the suite must be deterministic:
# pin tests to CPU x64. Builder runs opt into GPU explicitly outside the suite.
set_platform("cpu")
enable_x64()
