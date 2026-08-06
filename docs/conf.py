# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime

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
    "sphinxcontrib.autodoc_pydantic",
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

myst_heading_anchors = 5
myst_ref_domains = ["std", "py"]
myst_all_links_external = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.12", None),
    "typing_extensions": ("https://typing-extensions.readthedocs.io/en/latest", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

autodoc_mock_imports = ["torch", "safetensors", "transformers", "tvm_ffi"]
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": False,
    "member-order": "bysource",
}

autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_model_show_json = True
autodoc_pydantic_settings_show_json = False

# -- Other Options --------------------------------------------------------

templates_path = ["_templates"]

# Redirects for pages moved in the docs restructure. Keys are old docnames;
# values are the new locations relative to the old page.
redirects = {
    "tutorials/constrained_decoding": "../start/constrained_decoding.html",
    "tutorials/workflow_of_xgrammar": "../start/workflow_of_xgrammar.html",
    "tutorials/advanced_topics": "../start/workflow_of_xgrammar.html",
    "tutorials/engine_integration": "../using_xgrammar/engine_integration.html",
    "tutorials/json_generation": "../defining_structures/json_generation.html",
    "tutorials/ebnf_guided_generation": "../defining_structures/ebnf_grammar.html",
    "xgrammar_features/ebnf_grammar": "../defining_structures/ebnf_grammar.html",
    "xgrammar_features/runtime_safeguards": "../using_xgrammar/runtime_safeguards.html",
    "xgrammar_features/serialization": "../using_xgrammar/serialization.html",
    "xgrammar_features/lark_grammar": "../defining_structures/lark_grammar.html",
    "xgrammar_features/javascript_api": "../using_xgrammar/javascript_api.html",
    "structural_tag/structural_tag_api": "structural_tag.html",
}

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

suppress_warnings = ["misc.highlighting_failure", "autodoc.mocked_object"]

# A list of ignored prefixes for module index sorting.
# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

html_theme = "furo"

html_title = f"XGrammar {__version__}"

html_static_path = ["_static"]

html_theme_options = {
    "light_logo": "img/logo.png",
    "dark_logo": "img/logo_dark.svg",
    "source_repository": "https://github.com/mlc-ai/xgrammar",
    "source_branch": "main",
    "source_directory": "docs/",
}
