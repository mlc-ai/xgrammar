<div align="center" id="top">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./docs/_static/img/logo_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="./assets/logo.svg">
  <img src="./assets/logo.svg" alt="XGrammar" width="400">
</picture>

[![Documentation](https://img.shields.io/badge/docs-latest-green)](https://xgrammar.mlc.ai/docs/)
[![License](https://img.shields.io/badge/license-apache_2-blue)](https://github.com/mlc-ai/xgrammar/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/xgrammar)](https://pypi.org/project/xgrammar)
[![PyPI Downloads](https://static.pepy.tech/badge/xgrammar)](https://pepy.tech/projects/xgrammar)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mlc-ai/xgrammar)

**Efficient, Flexible and Portable Structured Generation**


[Get Started](#get-started) | [Documentation](https://xgrammar.mlc.ai/docs/) | [Blogpost](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar) | [Technical Report](https://arxiv.org/abs/2411.15100)

</div>

## News
- [2026/5] XGrammar-2 has been released! Check out our [blog](https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation) for more information.
- [2025/12] XGrammar has been officially integrated into [Mirai](https://github.com/trymirai/uzu)
- [2025/09] XGrammar has been officially integrated into [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)
- [2025/02] XGrammar has been officially integrated into [Modular's MAX](https://docs.modular.com/max/serve/structured-output)
- [2025/01] XGrammar has been officially integrated into [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).
- [2024/12] XGrammar has been officially integrated into [vLLM](https://github.com/vllm-project/vllm).
- [2024/12] We presented research talks on XGrammar at CMU, UC Berkeley, MIT, THU, SJTU, Ant Group, LMSys, Qingke AI, Camel AI. The slides can be found [here](https://docs.google.com/presentation/d/1iS7tu2EV4IKRWDaR0F3YD7ubrNqtGYUStSskceneelc/edit?usp=sharing).
- [2024/11] XGrammar has been officially integrated into [SGLang](https://github.com/sgl-project/sglang).
- [2024/11] XGrammar has been officially integrated into [MLC-LLM](https://github.com/mlc-ai/mlc-llm).
- [2024/11] We officially released XGrammar v0.1.0!

## Overview

XGrammar is an open-source library for efficient, flexible, and portable structured generation.

It leverages constrained decoding to ensure **100% structural correctness** of the output. It supports general context-free grammar to enable a broad range of structures, including **JSON**, **regex**, **custom context-free grammar**, etc.

XGrammar uses careful optimizations to achieve extremely low overhead in structured generation. It has achieved **near-zero overhead** in JSON generation, making it one of the fastest structured generation engines available.

XGrammar features **universal deployment**. It supports:
* **Platforms**: Linux, macOS, Windows
* **Hardware**: CPU, NVIDIA GPU, AMD GPU, Apple Silicon, TPU, etc.
* **Languages**: Python, C++, JavaScript, and Swift APIs
* **Models**: Qwen, Llama, DeepSeek, Phi, Gemma, etc.

XGrammar is very easy to integrate with LLM inference engines. It is the default structured generation backend for most LLM inference engines, including  [**vLLM**](https://github.com/vllm-project/vllm), [**SGLang**](https://github.com/sgl-project/sglang), [**TensorRT-LLM**](https://github.com/NVIDIA/TensorRT-LLM), and [**MLC-LLM**](https://github.com/mlc-ai/mlc-llm), as well as many other companies. You can also try out their structured generation modes!

## Get Started

Install XGrammar:
```bash
pip install xgrammar
```

For use with MPS on Apple Silicon, install with:
```bash
pip install "xgrammar[metal]"
```

Import XGrammar:
```python
import xgrammar as xgr
```

Please visit our [documentation](https://xgrammar.mlc.ai/docs/) to get started with XGrammar.
- [Installation](https://xgrammar.mlc.ai/docs/start/installation)
- [Quick start](https://xgrammar.mlc.ai/docs/start/quick_start)

## Contributing and governance

We welcome contributions of code, tests, documentation, bug reports, and reviews. See [CONTRIBUTING.md](CONTRIBUTING.md) to get started. All participants are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

[GOVERNANCE.md](GOVERNANCE.md) describes the current maintainers, technical decision process, and maintainer role changes. [CODEOWNERS](CODEOWNERS) lists the reviewers responsible for specific parts of the repository.

## Releases

XGrammar publishes versioned releases on [GitHub Releases](https://github.com/mlc-ai/xgrammar/releases) and Python packages on [PyPI](https://pypi.org/project/xgrammar/). Releases are made as needed when features, bug fixes, or compatibility updates are ready, without a fixed calendar schedule. Prereleases are identified separately from stable releases.

For each release, a maintainer selects a reviewed commit from `main` and coordinates the following:

1. Verify that the applicable automated tests and package builds pass for the release commit, and resolve failures that affect the release.
2. Prepare release notes describing features, fixes, compatibility changes, and known limitations.
3. Tag the release commit and publish a GitHub release. The [package workflow](.github/workflows/build_and_release.yaml) builds wheels and a source distribution and publishes them to PyPI when the GitHub release is published. Maintainers can also invoke that workflow manually.
4. Check that the published packages and release notes are available, and follow up on reported regressions with fixes or a subsequent release.

The Python package version is derived from Git tags; do not edit a version in
`pyproject.toml`. Use tags such as `v0.2.7`, `v0.2.7rc1`, or `v0.2.7.post1`.
`scripts/release_new_version.sh <tag>` checks that the current checkout is a clean,
up-to-date `main`, then creates and pushes an annotated tag. Publishing the GitHub
release starts package publication. To retry publication manually, run the package
workflow on the release tag; a manual run on a branch only builds artifacts.

Between tags, builds use the nearest reachable version tag and a development
suffix, for example `0.2.7.dev3+g<commit>` after three commits following `v0.2.6`.
Uncommitted changes also add a date suffix. The inferred development version does
not prescribe the next release number. Clone with full Git history and tags
(`git fetch --unshallow --tags` repairs a shallow clone). Published source
distributions retain their version in `PKG-INFO` and can build without Git metadata;
source copies without either Git history or distribution metadata cannot infer a
version. Installed package versions are available through
`importlib.metadata.version("xgrammar")` without running Git.

## Third-Party Bindings

- **Rust**: [xgrammar-rs](https://github.com/trymirai/xgrammar-rs) — Community Rust bindings for XGrammar.

## Collaborators

XGrammar has been widely adopted in industry, open-source projects, and academia. Our collaborators include:

<div align="center">

[<img src="https://raw.githubusercontent.com/mlc-ai/XGrammar-web-assets/refs/heads/main/repo/xai.png" height=50/>](https://x.ai/)
&emsp;
[<img src="https://raw.githubusercontent.com/mlc-ai/XGrammar-web-assets/refs/heads/main/repo/deepseek.png" height=50/>](https://www.deepseek.com/en/)
&emsp;
[<img src="https://raw.githubusercontent.com/mlc-ai/XGrammar-web-assets/refs/heads/main/repo/nvidia.svg" height=50/>](https://github.com/NVIDIA/TensorRT-LLM)
&emsp;
[<img src="https://raw.githubusercontent.com/mlc-ai/XGrammar-web-assets/refs/heads/main/repo/databricks.svg" height=50/>](https://www.databricks.com/)
&emsp;
[<img src="https://raw.githubusercontent.com/mlc-ai/XGrammar-web-assets/refs/heads/main/repo/meta.svg" height=50/>](https://about.meta.com/)
&emsp;
[<img src="https://raw.githubusercontent.com/mlc-ai/XGrammar-web-assets/refs/heads/main/repo/google.svg" height=50/>](https://about.google/)
&emsp;
[<img src="https://raw.githubusercontent.com/mlc-ai/XGrammar-web-assets/refs/heads/main/repo/perplexity.png" height=50/>](https://www.perplexity.ai/)
&emsp;
[<img src="https://raw.githubusercontent.com/mlc-ai/XGrammar-web-assets/refs/heads/main/repo/modular.svg" height=50/>](https://www.modular.com/)
&emsp;
[<img src="https://raw.githubusercontent.com/mlc-ai/XGrammar-web-assets/refs/heads/main/repo/sglang.png" height=50/>](https://github.com/sgl-project/sglang)
&emsp;
[<img src="https://raw.githubusercontent.com/mlc-ai/XGrammar-web-assets/refs/heads/main/repo/vllm.png" height=50/>](https://github.com/vllm-project/vllm)
&emsp;
[<img src="https://raw.githubusercontent.com/mlc-ai/XGrammar-web-assets/refs/heads/main/repo/mlc.jpeg" height=50/>](https://github.com/mlc-ai/mlc-llm)
&emsp;
[<span style="font-size:50px">WebLLM</span>](https://github.com/mlc-ai/web-llm)
&emsp;
[<img src="https://artifacts.trymirai.com/social/github/mirai.png" height=50/>](https://github.com/trymirai/uzu)

</div>

## Citation

If you find XGrammar useful in your research, please consider citing our papers:

```bibtex
@article{dong2024xgrammar,
  title={Xgrammar: Flexible and efficient structured generation engine for large language models},
  author={Dong, Yixin and Ruan, Charlie F and Cai, Yaxing and Lai, Ruihang and Xu, Ziyi and Zhao, Yilong and Chen, Tianqi},
  journal={Proceedings of Machine Learning and Systems 7},
  year={2024}
}
@inproceedings{10.1145/3786335.3813124,
  author = {Li, Linzhang and Dong, Yixin and Wang, Guanjie and Xu, Ziyi and Jiang, Alexander and Chen, Tianqi},
  title = {XGrammar-2: Dynamic and Efficient Structured Generation Engine for Agentic LLMs},
  year = {2026},
  isbn = {9798400724152},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3786335.3813124},
  booktitle = {Proceedings of the ACM Conference on AI and Agentic Systems},
  pages = {1009--1022},
  numpages = {14}
}
```
