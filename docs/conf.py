"""Sphinx configuration."""

from importlib.metadata import version as get_version

project = "tiptilt"
release = get_version("tiptilt")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

autoapi_dirs = ["../src"]
# Include the generated version module so AutoAPI can resolve __version__.
# AutoAPI already indexes fields; Napoleon should render their descriptions only.
napoleon_use_ivar = True
autodoc_typehints = "description"
exclude_patterns = ["_build", ".build", "Thumbs.db", ".DS_Store"]
language = "en"

myst_enable_extensions = ["amsmath", "dollarmath"]

html_theme = "sphinx_book_theme"
html_context = {"default_mode": "dark"}
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb"}

nb_execution_mode = "auto"
nb_execution_timeout = 120
nb_execution_raise_on_error = True
