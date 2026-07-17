"""Render Gemma-4 chat-template outputs without transformers.

Renders the official Gemma-4 chat templates with a jinja2 environment that
mirrors transformers' ``_compile_jinja_template``, so the output is
byte-identical to ``tokenizer.apply_chat_template``. This lets the alignment
test exercise Gemma-4 on any transformers version: loading the Gemma-4
tokenizer itself requires transformers >= 5.5, while this repo pins < 5.

Vendored templates (byte-exact copies of ``chat_template.jinja``, published
2026-07-09):

- ``chat_template_gemma4_e2b_it.jinja`` from google/gemma-4-E2B-it,
  snapshot 179516f0c449474fdc46f08f30ead5b11e178497
- ``chat_template_gemma4_31b_it.jinja`` from google/gemma-4-31B-it,
  snapshot b9ea41a2887d8607f594846523f94c6cc75ac8a4

The two templates are identical except that the 31B one pre-renders an empty
thought block in the generation prompt when thinking is disabled.

To refresh after an upstream template update::

    huggingface-cli download google/gemma-4-E2B-it chat_template.jinja
    huggingface-cli download google/gemma-4-31B-it chat_template.jinja

then copy the files over the vendored ones and re-run
``test_gemma_4_vendored_template_matches_official``.
"""

import os
from functools import lru_cache

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

_TEMPLATE_FILES = {
    "gemma4_e2b": "chat_template_gemma4_e2b_it.jinja",
    "gemma4_31b": "chat_template_gemma4_31b_it.jinja",
}


def _raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


@lru_cache(maxsize=None)
def _load_template(variant):
    # Mirrors transformers' _compile_jinja_template environment so rendering
    # is byte-identical to tokenizer.apply_chat_template.
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
    )
    env.globals["raise_exception"] = _raise_exception
    path = os.path.join(os.path.dirname(__file__), _TEMPLATE_FILES[variant])
    with open(path, encoding="utf-8") as f:
        return env.from_string(f.read())


class GemmaTemplateRenderer:
    """Drop-in stand-in for the Gemma-4 tokenizer in the alignment test.

    Only implements the ``apply_chat_template`` surface the test uses
    (``tokenize=False``), plus the ``eos_token`` attribute ``strip_eos``
    may read.
    """

    bos_token = "<bos>"
    eos_token = None

    def __init__(self, variant):
        if variant not in _TEMPLATE_FILES:
            raise ValueError(f"Unknown gemma-4 template variant: {variant}")
        self.variant = variant

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        tools=None,
        add_generation_prompt=False,
        enable_thinking=False,
        **kwargs,
    ):
        assert tokenize is False, "GemmaTemplateRenderer only supports tokenize=False"
        return _load_template(self.variant).render(
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            bos_token=self.bos_token,
            **kwargs,
        )
