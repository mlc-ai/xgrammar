<div align="center" id="top">

<img src="https://raw.githubusercontent.com/mlc-ai/xgrammar/main/assets/logo.svg" alt="logo" width="400" margin="10px"></img>

[![Documentation](https://img.shields.io/badge/docs-latest-green)](https://xgrammar.mlc.ai/docs/)
[![License](https://img.shields.io/badge/license-apache_2-blue)](https://github.com/mlc-ai/xgrammar/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/xgrammar)](https://pypi.org/project/xgrammar)
[![PyPI Downloads](https://static.pepy.tech/badge/xgrammar)](https://pepy.tech/projects/xgrammar)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mlc-ai/xgrammar)

**Efficient, Flexible and Portable Structured Generation**


[Get Started](#get-started) | [Documentation](https://xgrammar.mlc.ai/docs/) | [Blogpost](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar) | [Technical Report](https://arxiv.org/abs/2411.15100)

</div>

## News
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
* **Languages**: Python, C++, and JavaScript APIs
* **Models**: Qwen, Llama, DeepSeek, Phi, Gemma, etc.

XGrammar is very easy to integrate with LLM inference engines. It is the default structured generation backend for most LLM inference engines, including  [**vLLM**](https://github.com/vllm-project/vllm), [**SGLang**](https://github.com/sgl-project/sglang), [**TensorRT-LLM**](https://github.com/NVIDIA/TensorRT-LLM), and [**MLC-LLM**](https://github.com/mlc-ai/mlc-llm), as well as many other companies. You can also try out their structured generation modes!

## Get Started

Install XGrammar:
```bash
pip install xgrammar
```

Import XGrammar:
```python
import xgrammar as xgr
```

Please visit our [documentation](https://xgrammar.mlc.ai/docs/) to get started with XGrammar.
- [Installation](https://xgrammar.mlc.ai/docs/start/installation)
- [Quick start](https://xgrammar.mlc.ai/docs/start/quick_start)


## Collaborators

XGrammar has been widely adopted in industry, open-source projects, and academia. Our collaborators include:

<div align="center">

[<img src="https://raw.githubusercontent.com/mlc-ai/XGrammar-web-assets/refs/heads/main/repo/nvidia.svg" height=50/>](https://github.com/NVIDIA/TensorRT-LLM)
&emsp;
[<img src="https://raw.githubusercontent.com/mlc-ai/XGrammar-web-assets/refs/heads/main/repo/databricks.svg" height=50/>](https://www.databricks.com/)
&emsp;
[<img src="https://raw.githubusercontent.com/mlc-ai/XGrammar-web-assets/refs/heads/main/repo/xai.png" height=50/>](https://github.com/NVIDIA/TensorRT-LLM)
&emsp;
[<img src="https://raw.githubusercontent.com/mlc-ai/XGrammar-web-assets/refs/heads/main/repo/deepseek.png" height=50/>](https://github.com/NVIDIA/TensorRT-LLM)
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

</div>

## Citation

If you find XGrammar useful in your research, please consider citing our paper:

```bibtex
@article{dong2024xgrammar,
  title={Xgrammar: Flexible and efficient structured generation engine for large language models},
  author={Dong, Yixin and Ruan, Charlie F and Cai, Yaxing and Lai, Ruihang and Xu, Ziyi and Zhao, Yilong and Chen, Tianqi},
  journal={Proceedings of Machine Learning and Systems 7},
  year={2024}
}
```
