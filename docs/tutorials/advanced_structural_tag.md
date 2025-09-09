# Advanced Topics of the Structural Tag

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

### Parameters

- **tags** (`List[StructuralTagItem]`): The structural tags.
- **triggers** (`List[str]`): The triggers.

### Returns

- **grammar** (`Grammar`): The constructed grammar.

### Example

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
