# sphinx - configuration file
import sys
import os

project = "collasso"
copyright = "2026, Armin Rauschenberger"
author = "Armin Rauschenberger"
release = "0.1.0"

sys.path.insert(0, os.path.abspath("../../src"))

extensions = ["sphinx.ext.autodoc", "numpydoc", "myst_parser"]
source_suffix = [".rst", ".md"]

numpydoc_show_class_members = False

templates_path = ["_templates"]

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/rauschenberger/collasso",
    "show_toc_level": 2,
    "navigation_depth": 3,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}
