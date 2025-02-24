# -*- coding: utf-8 -*-
import os
import sys

import tlcpack_sphinx_addon

# -- General configuration ------------------------------------------------

os.environ["XGRAMMAR_BUILD_DOCS"] = "1"
sys.path.insert(0, os.path.abspath("../python"))
sys.path.insert(0, os.path.abspath("../"))
autodoc_mock_imports = ["torch"]

# General information about the project.
project = "XGrammar"
author = "XGrammar Contributors"
copyright = "2024, %s" % author

# Version information.

version = "0.1.0"
release = "0.1.0"

extensions = [
    "sphinx_tabs.tabs",
    "sphinx_toolbox.collapse",
    "sphinxcontrib.httpdomain",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_reredirects",
    "autodocsumm",
]

redirects = {}

source_suffix = [".rst"]

language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

# The theme is set by the make target
import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

templates_path = []

html_static_path = []

footer_copyright = "© 2024 XGrammar"
footer_note = " "

# html_logo = "_static/img/mlc-logo-with-text-landscape.svg"

html_theme_options = {"logo_only": False}

header_links = [
    ("Home", "https://xgrammar.mlc.ai/"),
    ("Github", "https://github.com/mlc-ai/xgrammar"),
]

header_dropdown = {"name": "Other Resources", "items": [("MLC Blog", "https://blog.mlc.ai/")]}

html_context = {
    "footer_copyright": footer_copyright,
    "footer_note": footer_note,
    "header_links": header_links,
    "header_dropdown": header_dropdown,
    "display_github": True,
    "github_user": "mlc-ai",
    "github_repo": "xgrammar",
    "github_version": "main/docs/",
    "theme_vcs_pageview_mode": "edit",
    # "header_logo": "/path/to/logo",
    # "header_logo_link": "",
    # "version_selecter": "",
}

import xgrammar

# add additional overrides
templates_path += [tlcpack_sphinx_addon.get_templates_path()]
html_static_path += [tlcpack_sphinx_addon.get_static_path()]
