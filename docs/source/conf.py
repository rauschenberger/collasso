# sphinx - configuration file

project = 'collasso'
copyright = '2026, Armin Rauschenberger'
author = 'Armin Rauschenberger'
release = '0.1.0'

import sys, os
sys.path.insert(0, os.path.abspath("../../src"))

extensions = ["sphinx.ext.autodoc",  "numpydoc"] # "sphinx.ext.autosummary",

#autosummary_generate = True
numpydoc_show_class_members = False

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "github_url": "https://github.com/rauschenberger/collasso",
    "show_toc_level": 2,
    "navigation_depth": 3,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}
