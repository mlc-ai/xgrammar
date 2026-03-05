import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string, _get_matcher_from_grammar
grammar = r"""root ::= [a-z]{1, 10}"""

grammar = xgr.Grammar.from_ebnf(grammar)
matcher = _get_matcher_from_grammar(grammar)