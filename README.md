<div align="center" id="top">

# XGrammar

[![Documentation](https://img.shields.io/badge/docs-latest-green)](https://xgrammar.mlc.ai/docs/)
[![License](https://img.shields.io/badge/license-apache_2-blue)](https://github.com/mlc-ai/xgrammar/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/xgrammar)](https://pypi.org/project/xgrammar)

**Efficient, Flexible and Portable Structured Generation**


[Get Started](#get-started) | [Documentation](https://xgrammar.mlc.ai/docs/) | [Blogpost](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar) | [Technical Report](https://arxiv.org/abs/2411.15100)

</div>

## News
- [2025/02] XGrammar has been officially integrated into [Modular's MAX](https://docs.modular.com/max/serve/structured-output)
- [2025/01] XGrammar has been officially integrated into [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).
- [2024/12] XGrammar has been officially integrated into [vLLM](https://github.com/vllm-project/vllm).
- [2024/12] We presented research talks on XGrammar at CMU Catalyst, Berkeley SkyLab, MIT HANLAB, THU IIIS, Ant Group, SGLang Meetup and Qingke AI. The slides can be found [here](https://docs.google.com/presentation/d/1iS7tu2EV4IKRWDaR0F3YD7ubrNqtGYUStSskceneelc/edit?usp=sharing).
- [2024/11] XGrammar has been officially integrated into [SGLang](https://github.com/sgl-project/sglang).
- [2024/11] XGrammar has been officially integrated into [MLC-LLM](https://github.com/mlc-ai/mlc-llm).
- [2024/11] We officially released XGrammar v0.1.0!

## Overview


XGrammar is an open-source library for efficient, flexible, and portable structured generation.
It supports general context-free grammar to enable a broad range of structures while bringing careful system optimizations to enable fast executions.
XGrammar features a minimal and portable C++ backend that can be easily integrated into multiple environments and frameworks,
and is co-designed with the LLM inference engine and enables zero-overhead structured generation in LLM inference.



## Get Started

Please visit our [documentation](https://xgrammar.mlc.ai/docs/) to get started with XGrammar.
- [Installation](https://xgrammar.mlc.ai/docs/start/install)
- [Quick start](https://xgrammar.mlc.ai/docs/start/quick_start)
