"""This test is adopted from test_builtin_grammar_json.py, but the grammar is parsed from
a unoptimized, non-simplified EBNF string. This is to test the robustness of the grammar matcher.
"""

import sys
import time
from typing import List

import pytest
import torch
from transformers import AutoTokenizer

import xgrammar as xgr
from xgrammar.testing import (
    _get_masked_tokens_from_bitmask,
    _get_matcher_from_grammar_and_tokenizer_info,
    _is_grammar_accept_string,
)


def test_simple():
    grammar_str = """root ::= rule1 rule2
rule1 ::= (rule2 | rule3) "a"
rule2 ::= "b"
rule3 ::= "c"
"""

    grammar = xgr.Grammar.from_ebnf(grammar_str)
    assert _is_grammar_accept_string(grammar, "bab")
    assert not _is_grammar_accept_string(grammar, "abb")
    assert _is_grammar_accept_string(grammar, "cab")


input_accepted_test_repetition = (
    ("aaa", True),
    ("abcbc", True),
    ("bcbcbcbcbc", True),
    ("bcbcbcbcbcbcbcb", True),
    ("d", False),
    ("aaaa", False),
)


@pytest.mark.parametrize("input, accepted", input_accepted_test_repetition)
def test_repetition(input: str, accepted: bool):
    grammar_str = """
        root ::= rule {2, 3}
        rule ::= ("a" | [bc] {4,})
    """
    grammar = xgr.Grammar.from_ebnf(grammar_str)
    assert _is_grammar_accept_string(grammar, input) == accepted


input_accepted_test_repetition_with_empty = (
    ("aaa", True),
    ("abcbc", True),
    ("bcbcbcbcbc", True),
    ("bcbcbcbcbcbcbcb", True),
    ("aaaa", False),
    ("", True),
    ("a", True),
    ("d", True),
)


@pytest.mark.parametrize("input, accepted", input_accepted_test_repetition_with_empty)
def test_repetition_with_empty(input: str, accepted: bool):
    grammar_str = """
        root ::= rule {2, 3} "d"?
        rule ::= ("a" | [bc] {4,}) | ""
    """
    grammar = xgr.Grammar.from_ebnf(grammar_str)
    assert _is_grammar_accept_string(grammar, input) == accepted


def test_utf8():
    # Test utf8-encoded string with EBNF grammar
    ebnf_grammar_str = "root ::= [，]+"

    grammar = xgr.Grammar.from_ebnf(ebnf_grammar_str)

    accepted_inputs = ["，", "，，，", "，，，，，，，，，，，，，，，，，，，，，，"]
    for input_str in accepted_inputs:
        assert _is_grammar_accept_string(grammar, input_str, print_time=True)


def test_custom_root_rule():
    json_grammar_simple_ebnf = r"""
root ::= basic_object
basic_any ::= basic_string | basic_object
basic_string ::= (([\"] basic_string_1 [\"]))
basic_string_1 ::= "" | [^"\\\r\n] basic_string_1 | "\\" escape basic_string_1
escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_object ::= "{" ("" | ws basic_string ws ":" ws basic_any ( ws "," ws basic_string ws ":" ws basic_any)*) ws "}"
ws ::= [ \n\t]*
"""
    grammar = xgr.Grammar.from_ebnf(json_grammar_simple_ebnf, root_rule_name="basic_string")
    assert _is_grammar_accept_string(grammar, r'"abc\r\n"')
    assert not _is_grammar_accept_string(grammar, r'{"name": "John" }')


json_grammar_ebnf = r"""
root ::= basic_array | basic_object
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= (([\"] basic_string_1 [\"]))
basic_string_1 ::= "" | [^"\\\x00-\x1F] basic_string_1 | "\\" escape basic_string_1
escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= "[" ("" | ws basic_any (ws "," ws basic_any)*) ws "]"
basic_object ::= "{" ("" | ws basic_string ws ":" ws basic_any ( ws "," ws basic_string ws ":" ws basic_any)*) ws "}"
ws ::= [ \n\t]*
"""
json_grammar = xgr.Grammar.from_ebnf(json_grammar_ebnf)


json_input_accepted = [
    '{"name": "John"}',
    '{ "name" : "John" }',
    "{}",
    "[]",
    '{"name": "Alice", "age": 30, "city": "New York"}',
    '{"name": "Mike", "hobbies": ["reading", "cycling", "hiking"]}',
    '{"name": "Emma", "address": {"street": "Maple Street", "city": "Boston"}}',
    '[{"name": "David"}, {"name": "Sophia"}]',
    (
        '{"name": "William", "age": null, "married": true, "children": ["Liam", "Olivia"],'
        ' "hasPets": false}'
    ),
    (
        '{"name": "Olivia", "contact": {"email": "olivia@example.com", "address": '
        '{"city": "Chicago", "zipcode": "60601"}}}'
    ),
    (
        '{"name": "Liam", "skills": ["Java", "Python"], "experience": '
        '[{"company": "CompanyA", "years": 5}, {"company": "CompanyB", "years": 3}]}'
    ),
    (
        '{"person": {"name": "Ethan", "age": 40}, "education": {"degree": "Masters", '
        '"university": "XYZ University"}, "work": [{"company": "ABC Corp", "position": '
        '"Manager"}, {"company": "DEF Corp", "position": "Senior Manager"}]}'
    ),
    (
        '{"name": "Charlotte", "details": {"personal": {"age": 35, "hobbies": ["gardening", '
        '"painting"]}, "professional": {"occupation": "Engineer", "skills": '
        '["CAD", "Project Management"], "projects": [{"name": "Project A", '
        '"status": "Completed"}, {"name": "Project B", "status": "In Progress"}]}}}'
    ),
]


@pytest.mark.parametrize("json_input_accepted", json_input_accepted)
def test_json_accept(json_input_accepted: str):
    assert _is_grammar_accept_string(json_grammar, json_input_accepted)


json_input_refused = (
    r'{ name: "John" }',
    r'{ "name": "John" } ',  # trailing space is not accepted
    r'{ "name": "John", "age": 30, }',
    r'{ "name": "John", "address": { "street": "123 Main St", "city": "New York" }',
    r'{ "name": "John", "age": 30, "hobbies": ["reading", "traveling",], }',
    r'{ "name": "John", "age": 30.5.7 }',
    r'{ "name": "John, "age": 30, "hobbies": ["reading", "traveling"] }',
    (
        r'{ "name": "John", "age": 30, "hobbies": ["reading", { "type": "outdoor", "list": '
        r'["hiking", "swimming",]}] }'
    ),
    r'{ "name": "John", "age": 30, "status": "\P\J" }',
    (
        r'{ "name": "John", "age": 30, "hobbies": ["reading", "traveling"], "address": '
        r'{ "street": "123 Main St", "city": "New York", "coordinates": { "latitude": 40.7128, '
        r'"longitude": -74.0060 }}}, "work": { "company": "Acme", "position": "developer" }}'
    ),
)


@pytest.mark.parametrize("json_input_refused", json_input_refused)
def test_json_refuse(json_input_refused: str):
    assert not _is_grammar_accept_string(json_grammar, json_input_refused)


json_input_pressure = (
    # Extra long string: 1k chars
    (
        '["Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer nec odio. Praesent '
        "libero. Sed cursus ante dapibus diam. Sed nisi. Nulla quis sem at nibh elementum "
        "imperdiet. Duis sagittis ipsum. Praesent mauris. Fusce nec tellus sed augue semper "
        "porta. Mauris massa. Vestibulum lacinia arcu eget nulla. Class aptent taciti sociosqu "
        "ad litora torquent per conubia nostra, per inceptos himenaeos. Curabitur sodales ligula "
        "in libero. Sed dignissim lacinia nunc. Curabitur tortor. Pellentesque nibh. Aenean quam. "
        "In scelerisque sem at dolor. Maecenas mattis. Sed convallis tristique sem. Proin ut "
        "ligula vel nunc egestas porttitor. Morbi lectus risus, iaculis vel, suscipit quis, "
        "luctus non, massa. Fusce ac turpis quis ligula lacinia aliquet. Mauris ipsum. Nulla "
        "metus metus, ullamcorper vel, tincidunt sed, euismod in, nibh. Quisque volutpat "
        "condimentum velit. Class aptent taciti sociosqu ad litora torquent per conubia nostra, "
        "per inceptos himenaeos. Nam nec ante. Sed lacinia, urna non tincidunt mattis, tortor "
        "neque adipiscing diam, a cursus ipsum ante quis turpis. Nulla facilisi. Ut fringilla. "
        "Suspendisse potenti. Nunc feugiat mi a tellus consequat imperdiet. Vestibulum sapien. "
        "Proin quam. Etiam ultrices. Suspendisse in justo eu magna luctus suscipit. Sed lectus. "
        "Integer euismod lacus luctus magna. Quisque cursus, metus vitae pharetra auctor, sem "
        'massa mattis sem, at interdum magna augue eget diam."]'
    ),
    # long and complex json: 3k chars
    (
        r"""{
    "web-app": {
    "servlet": [
        {
        "servlet-name": "cofaxCDS",
        "servlet-class": "org.cofax.cds.CDSServlet",
        "init-param": {
            "configGlossary:installationAt": "Philadelphia, PA",
            "configGlossary:adminEmail": "ksm@pobox.com",
            "configGlossary:poweredBy": "Cofax",
            "configGlossary:poweredByIcon": "/images/cofax.gif",
            "configGlossary:staticPath": "/content/static",
            "templateProcessorClass": "org.cofax.WysiwygTemplate",
            "templateLoaderClass": "org.cofax.FilesTemplateLoader",
            "templatePath": "templates",
            "templateOverridePath": "",
            "defaultListTemplate": "listTemplate.htm",
            "defaultFileTemplate": "articleTemplate.htm",
            "useJSP": false,
            "jspListTemplate": "listTemplate.jsp",
            "jspFileTemplate": "articleTemplate.jsp",
            "cachePackageTagsTrack": 200,
            "cachePackageTagsStore": 200,
            "cachePackageTagsRefresh": 60,
            "cacheTemplatesTrack": 100,
            "cacheTemplatesStore": 50,
            "cacheTemplatesRefresh": 15,
            "cachePagesTrack": 200,
            "cachePagesStore": 100,
            "cachePagesRefresh": 10,
            "cachePagesDirtyRead": 10,
            "searchEngineListTemplate": "forSearchEnginesList.htm",
            "searchEngineFileTemplate": "forSearchEngines.htm",
            "searchEngineRobotsDb": "WEB-INF/robots.db",
            "useDataStore": true,
            "dataStoreClass": "org.cofax.SqlDataStore",
            "redirectionClass": "org.cofax.SqlRedirection",
            "dataStoreName": "cofax",
            "dataStoreDriver": "com.microsoft.jdbc.sqlserver.SQLServerDriver",
            "dataStoreUrl": "jdbc:microsoft:sqlserver://LOCALHOST:1433;DatabaseName=goon",
            "dataStoreUser": "sa",
            "dataStorePassword": "dataStoreTestQuery",
            "dataStoreTestQuery": "SET NOCOUNT ON;select test='test';",
            "dataStoreLogFile": "/usr/local/tomcat/logs/datastore.log",
            "dataStoreInitConns": 10,
            "dataStoreMaxConns": 100,
            "dataStoreConnUsageLimit": 100,
            "dataStoreLogLevel": "debug",
            "maxUrlLength": 500
        }
        },
        {
        "servlet-name": "cofaxEmail",
        "servlet-class": "org.cofax.cds.EmailServlet",
        "init-param": {
            "mailHost": "mail1",
            "mailHostOverride": "mail2"
        }
        },
        {
        "servlet-name": "cofaxAdmin",
        "servlet-class": "org.cofax.cds.AdminServlet"
        },
        {
        "servlet-name": "fileServlet",
        "servlet-class": "org.cofax.cds.FileServlet"
        },
        {
        "servlet-name": "cofaxTools",
        "servlet-class": "org.cofax.cms.CofaxToolsServlet",
        "init-param": {
            "templatePath": "toolstemplates/",
            "log": 1,
            "logLocation": "/usr/local/tomcat/logs/CofaxTools.log",
            "logMaxSize": "",
            "dataLog": 1,
            "dataLogLocation": "/usr/local/tomcat/logs/dataLog.log",
            "dataLogMaxSize": "",
            "removePageCache": "/content/admin/remove?cache=pages&id=",
            "removeTemplateCache": "/content/admin/remove?cache=templates&id=",
            "fileTransferFolder": "/usr/local/tomcat/webapps/content/fileTransferFolder",
            "lookInContext": 1,
            "adminGroupID": 4,
            "betaServer": true
        }
        }
    ],
    "servlet-mapping": {
        "cofaxCDS": "/",
        "cofaxEmail": "/cofaxutil/aemail/*",
        "cofaxAdmin": "/admin/*",
        "fileServlet": "/static/*",
        "cofaxTools": "/tools/*"
    },
    "taglib": {
        "taglib-uri": "cofax.tld",
        "taglib-location": "/WEB-INF/tlds/cofax.tld"
    }
    }
}"""
    ),
)


@pytest.mark.parametrize("json_input_pressure", json_input_pressure)
def test_json_pressure(json_input_pressure: str):
    assert _is_grammar_accept_string(json_grammar, json_input_pressure, print_time=True)


tokenizer_path__input_str__expected_rejected_sizes = [
    (
        # short test
        "meta-llama/Llama-2-7b-chat-hf",
        '{"id": 1,"name": "Example"}',
        [
            # fmt: off
            31989, 31912, 270, 270, 270, 31973, 31846, 31846, 31948, 31915, 270, 270, 270, 270,
            270, 31973, 31846, 31846, 263, 263, 263, 263, 263, 263, 263, 263, 31974, 31999,
            # fmt: on
        ],
    ),
    (
        # long test
        "meta-llama/Llama-2-7b-chat-hf",
        """{
"id": 1,
"na": "ex",
"ac": true,
"t": ["t1", "t2"],
"ne": {"lv2": {"val": "dp"}, "arr": [1, 2, 3]},
"res": "res"
}""",
        [
            # fmt: off
            31989, 31912, 31912, 270, 270, 270, 31973, 31846, 31846, 31948, 31915, 31915, 270, 270,
            270, 31973, 31846, 31846, 263, 263, 263, 31974, 31915, 31915, 270, 270, 270, 31973,
            31846, 31846, 31997, 31997, 31998, 31974, 31915, 31915, 270, 270, 31973, 31846, 31846,
            31840, 262, 262, 262, 31969, 31846, 31846, 262, 262, 262, 31969, 31974, 31915, 31915,
            270, 270, 270, 31973, 31846, 31846, 31908, 270, 270, 270, 270, 31973, 31846, 31846,
            31906, 270, 270, 270, 270, 31973, 31846, 31846, 262, 262, 262, 31968, 31970, 31915,
            31915, 270, 270, 270, 270, 31973, 31846, 31846, 31840, 31943, 31846, 31846, 31943,
            31846, 31846, 31943, 31970, 31974, 31915, 31915, 270, 270, 270, 270, 31973, 31846,
            31846, 263, 263, 263, 263, 31974, 31974, 31999,
            # fmt: on
        ],
    ),
    (
        # test for llama 3
        "meta-llama/Meta-Llama-3-8B-Instruct",
        '{"id": 1,"name": "Example哈哈"}',
        [
            # fmt: off
            128235, 127497, 4744, 4744, 4744, 127849, 126399, 126399, 126760, 127499, 4744, 4744,
            4744, 4744, 4744, 127849, 126399, 126399, 4694, 4694, 4694, 4694, 4694, 4694, 4694,
            4694, 128066, 128111, 4694, 128066, 128111, 4694, 127873, 128255,
            # fmt: on
        ],
    ),
]


@pytest.mark.hf_token_required
@pytest.mark.parametrize(
    "tokenizer_path, input_str, expected_rejected_sizes",
    tokenizer_path__input_str__expected_rejected_sizes,
)
def test_fill_next_token_bitmask(
    tokenizer_path: str, input_str: str, expected_rejected_sizes: List[int]
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiler = xgr.GrammarCompiler(tokenizer_info)

    time_start = time.monotonic_ns()
    matcher = xgr.GrammarMatcher(compiler.compile_grammar(json_grammar_ebnf))
    time_end = time.monotonic_ns()
    print(f"Time to init GrammarMatcher: {(time_end - time_start) / 1e3} us")

    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logits_gpu = torch.zeros(tokenizer_info.vocab_size, dtype=torch.float32, device=device)

    input_bytes = input_str.encode("utf-8")

    for i, c in enumerate(input_bytes):
        # 1. fill_next_token_bitmask
        time_start = time.monotonic_ns()
        matcher.fill_next_token_bitmask(token_bitmask)
        time_end = time.monotonic_ns()
        print(f"Time to fill_next_token_bitmask: {(time_end - time_start) / 1e3} us")

        # 2. Correctness verification
        rejected_token_ids = _get_masked_tokens_from_bitmask(
            token_bitmask, tokenizer_info.vocab_size
        )
        assert len(rejected_token_ids) == expected_rejected_sizes[i]

        # 3. apply_token_bitmask_inplace
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_start = time.monotonic_ns()
        xgr.apply_token_bitmask_inplace(logits_gpu, token_bitmask.to(device))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_end = time.monotonic_ns()
        print(f"Time to apply_token_bitmask_inplace: {(time_end - time_start) / 1e3} us")

        # 4. accept_string
        print("Accepting char:", bytes([c]))
        time_start = time.monotonic_ns()
        assert matcher.accept_string(bytes([c]))
        time_end = time.monotonic_ns()
        print(f"Time to accept_token: {(time_end - time_start) / 1e3} us")

    # 5. Final correctness verification
    matcher.fill_next_token_bitmask(token_bitmask)
    rejected_token_ids = _get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size)
    assert len(rejected_token_ids) == expected_rejected_sizes[-1]


def test_nullable_grammar():
    grammar_with_nullable_rules = """
    root ::= rule1 | (rule1 rule1 rule1 rule3)+
    rule1 ::= rule2
    rule2 ::= [0-9]*
    rule3 ::= [a-z]
"""
    test_string = ["abc12312398014a", ""]

    for s in test_string:
        assert _is_grammar_accept_string(grammar_with_nullable_rules, s)


def test_predict_complete():
    # Test complex prediction and completion with EBNF grammar.
    mixed_grammar_str = """root ::= rule1 [0-9]?
    rule1 ::= rule2 [0-9]? | rule4 [0-9]?
    rule2 ::= rule3 [0-9]? | rule2 [0-9]? | rule1 [0-9]?
    rule3 ::= rule4 [0-9]? | rule5 [0-9]?
    rule4 ::= rule5 [0-9]? | rule6 [0-9]?
    rule5 ::= rule6 [0-9]? | rule7 [0-9]? | rule8 [0-9]?
    rule6 ::= rule7 [0-9]? | rule1 [0-9]?
    rule7 ::= rule8 [0-9]? | rule9 [0-9]?
    rule8 ::= rule9 [0-9]? | rule7 [0-9]?
    rule9 ::= [0-9]?
    """

    grammar = xgr.Grammar.from_ebnf(mixed_grammar_str)
    input_str = ""
    for i in range(10):
        assert _is_grammar_accept_string(grammar, input_str)
        input_str += "0"
    assert _is_grammar_accept_string(grammar, input_str)

    # Test right recursion
    right_recursion_grammar = "root ::= [a-z] root | [a-z]"

    accept_strings = ["a", "ab", "abc", "abcd", "abcde"]
    reject_strings = ["", "1", "a1", "ab1", "abc1"]
    for accept_string in accept_strings:
        assert _is_grammar_accept_string(right_recursion_grammar, accept_string)
    for reject_string in reject_strings:
        assert not _is_grammar_accept_string(right_recursion_grammar, reject_string)

    # Test the mixture of right recursion and other rules
    mixed_grammar_str = """root ::= rule1
    rule1 ::= "{" rule2 | ""
    rule2 ::= root "}"
    """
    test_strings = {"", "{}", "{{}}", "{{{}}}", "{{{{}}}}", "{{{{{}}}}}"}
    rejected_strings = {"{", "{}{}", "{{{{}", "{{}}}", "{{{{{}}}}}}"}

    for test_string in test_strings:
        assert _is_grammar_accept_string(mixed_grammar_str, test_string)
    for rejected_string in rejected_strings:
        assert not _is_grammar_accept_string(mixed_grammar_str, rejected_string)


def test_advance():
    # Test complex Advance and completion with EBNF grammar.
    ebnf_grammar_str = """root ::= rule1
    rule1 ::= [a] | [a-b] | [a-c]* | "a" | "aaaaaaaaaaaaaaaaaaa"
    """
    grammar = xgr.Grammar.from_ebnf(ebnf_grammar_str)
    for i in range(10):
        input_str = "a" * i
        assert _is_grammar_accept_string(grammar, input_str)


def test_character_class_star_utf8():
    ebnf_grammar_str = """root ::= [^0-9]*"""
    test_string = "worldせかい世界"
    assert _is_grammar_accept_string(ebnf_grammar_str, test_string)


@pytest.mark.hf_token_required
def test_not_neighbour_character_class():
    raw_grammar = "root ::= [a-cx-z]*"
    tokenizer_path = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    grammar = xgr.Grammar.from_ebnf(raw_grammar)
    matcher = _get_matcher_from_grammar_and_tokenizer_info(grammar, tokenizer_info)
    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    matcher.fill_next_token_bitmask(token_bitmask)
    rejected_token_ids = _get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size)
    assert len(rejected_token_ids) == 31933


def test_nfa():
    grammar_str = """
root ::= rule1 | rule2 | rule3
rule1 ::= "abc" | ""
rule2 ::= "abd" | ""
rule3 ::= [a-n] [b-c] "x" | ""
"""
    grammar = xgr.Grammar.from_ebnf(grammar_str)
    assert _is_grammar_accept_string(grammar, "abc")
    assert _is_grammar_accept_string(grammar, "abx")
    assert _is_grammar_accept_string(grammar, "ccx")
    assert not _is_grammar_accept_string(grammar, "abb")
    assert not _is_grammar_accept_string(grammar, "ad")


if __name__ == "__main__":
    pytest.main(sys.argv)
