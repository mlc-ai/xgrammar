Builtin Structural Tag
================================

.. currentmodule:: xgrammar.builtin_structural_tag

This page contains the API reference for the structural tag template function. For its usage, see
:doc:`Tool Calling and Reasoning <../../structural_tag/tool_calling_and_reasoning>`.

.. autofunction:: get_model_structural_tag

.. autofunction:: register_model_structural_tag

.. data:: get_builtin_structural_tag

   Deprecated alias for :func:`get_model_structural_tag`.

Registered Structural Tag Templates
-----------------------------------

The following model-specific APIs are registered via
:func:`register_model_structural_tag`.

.. autofunction:: get_llama_structural_tag

.. autofunction:: get_kimi_structural_tag

.. autofunction:: get_deepseek_r1_structural_tag

.. autofunction:: get_deepseek_v3_1_structural_tag

.. autofunction:: get_qwen_3_coder_structural_tag

.. autofunction:: get_qwen_3_5_structural_tag

.. autofunction:: get_qwen_3_structural_tag

.. autofunction:: get_harmony_structural_tag

.. autofunction:: get_deepseek_v3_2_structural_tag

.. autofunction:: get_minimax_structural_tag

.. autofunction:: get_glm_4_7_structural_tag

.. autofunction:: get_deepseek_v4_structural_tag
