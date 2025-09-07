# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime

import tlcpack_sphinx_addon
import tomli

# -- General configuration ------------------------------------------------

os.environ["XGRAMMAR_BUILD_DOCS"] = "1"
sys.path.insert(0, os.path.abspath("../python"))
sys.path.insert(0, os.path.abspath("../"))

# Load version from pyproject.toml
with open("../pyproject.toml", "rb") as f:
    pyproject_data = tomli.load(f)
__version__ = pyproject_data["project"]["version"]

project = "XGrammar"
author = "XGrammar Contributors"
copyright = f"2024-{datetime.now().year}, {author}"

version = __version__
release = __version__

# -- Extensions and extension configurations --------------------------------

extensions = [
    "myst_parser",
    "nbsphinx",
    "autodocsumm",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_reredirects",
    "sphinx_tabs.tabs",
    "sphinx_toolbox.collapse",
    "sphinxcontrib.httpdomain",
    "sphinxcontrib.mermaid",
]

nbsphinx_allow_errors = True
nbsphinx_execute = "never"

autosectionlabel_prefix_document = True
nbsphinx_allow_directives = True

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
    "html_image",
    "linkify",
    "substitution",
]

myst_heading_anchors = 3
myst_ref_domains = ["std", "py"]
myst_all_links_external = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.12", None),
    "typing_extensions": ("https://typing-extensions.readthedocs.io/en/latest", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

autodoc_mock_imports = ["torch", "safetensors", "transformers"]
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": False,
    "member-order": "bysource",
}

# -- Other Options --------------------------------------------------------

templates_path = []

redirects = {}

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

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

html_static_path = ["_static"]

# Add custom CSS files to fix text selection issues
html_css_files = ["css/fix_text_selection.css"]

footer_copyright = "© 2024 XGrammar"
footer_note = " "

# html_logo = "_static/img/logo.png"
# html_theme_options = {"logo_only": True}

header_links = [
    ("Home", "https://xgrammar.mlc.ai/"),
    ("Docs", "https://xgrammar.mlc.ai/docs/"),
    ("Github", "https://github.com/mlc-ai/xgrammar"),
    ("Blog", "https://blog.mlc.ai/"),
]

html_context = {
    "footer_copyright": footer_copyright,
    "footer_note": footer_note,
    "header_links": header_links,
    "display_github": True,
    "github_user": "mlc-ai",
    "github_repo": "xgrammar",
    "github_version": "main/docs/",
    "theme_vcs_pageview_mode": "edit",
    # Set the logo in left sidebar
    "logo": "img/logo.png",
    "theme_logo_only": True,
    # "header_logo": "_static/img/logo.png",
    # "header_logo_link": "",
    # "version_selecter": "",
}

# add additional overrides
templates_path += [tlcpack_sphinx_addon.get_templates_path()]
html_static_path += [tlcpack_sphinx_addon.get_static_path()]
