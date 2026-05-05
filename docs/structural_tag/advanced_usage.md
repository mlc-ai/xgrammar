# Advanced Topics of Structural Tag

## Automatic End Detection

Certain "unlimited" formats — `any_text`, `any_tokens`, `triggered_tags`, and `token_triggered_tags` — can consume an unbounded amount of output. To know when to stop, the structural tag compiler automatically detects the **end condition** of the enclosing `tag` and adds it to the format's internal exclude set.

The detection works by walking up to the nearest enclosing `tag`:

- If the tag's `end` is a **string** (or list of strings), the end strings are added to the exclude set of string-level unlimited formats (`any_text`, `triggered_tags`).
- If the tag's `end` is a **`token` format**, the end token ID is added to the exclude set of token-level unlimited formats (`any_tokens`, `token_triggered_tags`, `exclude_token`).

Currently only **string–string** and **token–token** pairs are detected. Cross-level detection (e.g. a token end for a string-level format) is not supported. If you need additional exclusions beyond what automatic detection provides, specify them explicitly via the format's own exclude field (`excludes` for `any_text`, `exclude_tokens` for `exclude_token`/`any_tokens`/`token_triggered_tags`).

---

## Deprecated API: `Grammar.from_structural_tag(tags, triggers)`

**The deprecated API is still available for backward compatibility. However, it is recommended to use the new API instead.**

Create a grammar from structural tags. The structural tag handles the dispatching of different grammars based on the tags and triggers: it initially allows any output, until a trigger is encountered, then dispatch to the corresponding tag; when the end tag is encountered, the grammar will allow any following output, until the next trigger is encountered.

The tags parameter is used to specify the output pattern. It is especially useful for LLM function calling, where the pattern is:
`<function=func_name>{"arg1": ..., "arg2": ...}</function>`.
This pattern consists of three parts: a begin tag (`<function=func_name>`), a parameter list according to some schema (`{"arg1": ..., "arg2": ...}`), and an end tag (`</function>`). This pattern can be described in a StructuralTagItem with a begin tag, a schema, and an end tag. The structural tag is able to handle multiple such patterns by passing them into multiple tags.

The triggers parameter is used to trigger the dispatching of different grammars. The trigger should be a prefix of a provided begin tag. When the trigger is encountered, the corresponding tag should be used to constrain the following output. There can be multiple tags matching the same trigger. Then if the trigger is encountered, the following output should match one of the tags. For example, in function calling, the triggers can be `["<function="]`. Then if `"<function="` is encountered, the following output must match one of the tags (e.g. `<function=get_weather>{"city": "Beijing"}</function>`).

The correspondence of tags and triggers is automatically determined: all tags with the same trigger will be grouped together. User should make sure any trigger is not a prefix of another trigger: then the correspondence of tags and triggers will be ambiguous.

To use this grammar in grammar-guided generation, the GrammarMatcher constructed from structural tag will generate a mask for each token. When the trigger is not encountered, the mask will likely be all-1 and not have to be used (fill_next_token_bitmask returns False, meaning no token is masked). When a trigger is encountered, the mask should be enforced (fill_next_token_bitmask will return True, meaning some token is masked) to the output logits.

The benefit of this method is the token boundary between tags and triggers is automatically handled. The user does not need to worry about the token boundary.

### Parameters (deprecated)

- **tags** (`List[StructuralTagItem]`): The structural tags.
- **triggers** (`List[str]`): The triggers.

### Returns (deprecated)

- **grammar** (`Grammar`): The constructed grammar.

### Example (deprecated)

```python
from pydantic import BaseModel
from typing import List
from xgrammar import Grammar, StructuralTagItem

class Schema1(BaseModel):
    arg1: str
    arg2: int

class Schema2(BaseModel):
    arg3: float
    arg4: List[str]

tags = [
    StructuralTagItem(begin="<function=f>", schema=Schema1, end="</function>"),
    StructuralTagItem(begin="<function=g>", schema=Schema2, end="</function>"),
]
triggers = ["<function="]
grammar = Grammar.from_structural_tag(tags, triggers)
```
