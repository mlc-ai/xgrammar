# Building Docs

XGrammar uses Sphinx to build the documentation, and the documentation is hosted on GitHub Pages.
The document can be written in Markdown (`.md`, preferred) or reStructuredText (`.rst`).

## Building Docs Locally

Install the dependencies:

```bash
# For non-Debian-based systems or Conda environments, use your package manager to install ruby.
sudo apt update
sudo apt install ruby-full
python -m pip install -r docs/requirements.txt
gem install jekyll jekyll-remote-theme
```

Build the docs locally:

```bash
bash scripts/local_deploy_site.sh
```

This will build the website and the docs, and host them locally at `http://localhost:8888`.

## Deploying Docs on GitHub Pages

The documentation is built and deployed automatically when you merge your changes into the `main` branch.
The workflow is defined in [`.github/workflows/documentation.yaml`](https://github.com/mlc-ai/xgrammar/tree/v0.1.19/.github/workflows/documentation.yaml).

The docs will be build locally, uploaded to [`xgrammar/gh-pages`](https://github.com/mlc-ai/xgrammar/tree/gh-pages) and then deployed to GitHub Pages.

## Best Practices for Writing Docs

When adding new features to XGrammar, please update the documentation accordingly.

Each time you make changes to the docs, you need to build the docs locally to see the changes and
make sure the changes are correct.

When referencing a code in the repository, make sure you are referring to a specific release version
of the code, instead of the main branch.
