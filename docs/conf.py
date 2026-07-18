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
autoapi_ignore = ["**/*version.py"]
myst_enable_extensions = ["amsmath", "dollarmath"]

html_theme = "sphinx_book_theme"
html_context = {"default_mode": "dark"}
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb"}

nb_execution_mode = "auto"
nb_execution_timeout = 120
nb_execution_raise_on_error = True
