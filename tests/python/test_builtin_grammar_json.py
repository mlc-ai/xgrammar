"""This test uses the optimized JSON grammar provided by the grammar library."""

import sys
import time
from typing import List, Optional

import pytest
from transformers import AutoTokenizer

from xgrammar import BNFGrammar, BuiltinGrammar, GrammarMatcher

json_grammar = BuiltinGrammar.json()


def match_complete_string(grammar: BNFGrammar, input_str: str) -> bool:
    matcher = GrammarMatcher(grammar, terminate_without_stop_token=True)
    can_accept = matcher.accept_string(input_str)
    can_terminate = matcher.is_terminated()
    return can_accept and can_terminate


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
    assert match_complete_string(json_grammar, json_input_accepted)


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
    assert not match_complete_string(json_grammar, json_input_refused)


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
    assert match_complete_string(json_grammar, json_input_pressure)


tokenizer_path__input_str__expected_rejected_sizes = [
    (
        # short test
        "meta-llama/Llama-2-7b-chat-hf",
        '{"id": 1,"name": "Example"}',
        [
            # fmt: off
            31989, 31912, 272, 272, 272, 31973, 31846, 31846, 31948, 31915, 272, 272, 272, 272,
            272, 31973, 31846, 31846, 265, 265, 265, 265, 265, 265, 265, 265, 31974, 31999,
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
            31989, 31912, 31912, 272, 272, 272, 31973, 31846, 31846, 31948, 31915, 31915, 272, 272,
            272, 31973, 31846, 31846, 265, 265, 265, 31974, 31915, 31915, 272, 272, 272, 31973,
            31846, 31846, 31997, 31997, 31998, 31974, 31915, 31915, 272, 272, 31973, 31846, 31846,
            31840, 264, 264, 264, 31969, 31846, 31846, 264, 264, 264, 31969, 31974, 31915, 31915,
            272, 272, 272, 31973, 31846, 31846, 31908, 272, 272, 272, 272, 31973, 31846, 31846,
            31906, 272, 272, 272, 272, 31973, 31846, 31846, 264, 264, 264, 31968, 31970, 31915,
            31915, 272, 272, 272, 272, 31973, 31846, 31846, 31840, 31943, 31846, 31846, 31943,
            31846, 31846, 31943, 31970, 31974, 31915, 31915, 272, 272, 272, 272, 31973, 31846,
            31846, 265, 265, 265, 265, 31974, 31974, 31999,
            # fmt: on
        ],
    ),
    (
        # test for llama 3
        "meta-llama/Meta-Llama-3-8B-Instruct",
        '{"id": 1,"name": "Example哈哈"}',
        [
            # fmt: off
            128235, 127497, 5002, 5002, 5002, 127849, 126399, 126399, 126760, 127499, 5002, 5002,
            5002, 5002, 5002, 127849, 126399, 126399, 4952, 4952, 4952, 4952, 4952, 4952, 4952,
            4952, 128066, 128111, 4952, 128066, 128111, 4952, 127873, 128254,
            # fmt: on
        ],
    ),
]


@pytest.mark.parametrize(
    "tokenizer_path, input_str, expected_rejected_sizes",
    tokenizer_path__input_str__expected_rejected_sizes,
)
def test_get_next_rejected_tokens(
    tokenizer_path: str,
    input_str: str,
    expected_rejected_sizes: Optional[List[int]],
):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
    )
    matcher = GrammarMatcher(json_grammar, tokenizer)
    input_bytes = input_str.encode("utf-8")
    rejected_sizes = []

    for i, c in enumerate(input_bytes):
        time_start = time.monotonic_ns()
        bitmask = matcher.get_next_token_bitmask()
        time_mid = time.monotonic_ns()
        rejected_token_ids = GrammarMatcher.get_rejected_tokens_from_bitmask(
            bitmask, matcher.mask_vocab_size
        )
        time_end = time.monotonic_ns()
        print(f"Time to get_next_token_bitmask: {(time_mid - time_start) / 1e3} us")
        print(f"Time to get_rejected_tokens_from_bitmask: {(time_end - time_mid) / 1e3} us")
        rejected_sizes.append(len(rejected_token_ids))
        if expected_rejected_sizes is not None:
            assert rejected_sizes[-1] == expected_rejected_sizes[i], (
                rejected_sizes[-1],
                expected_rejected_sizes[i],
            )
        print("Accepting char:", bytes([c]))
        time_start = time.monotonic_ns()
        assert matcher.accept_string(bytes([c]))
        time_end = time.monotonic_ns()
        print(f"Time to accept_token: {(time_end - time_start) / 1e3} us")

    bitmask = matcher.get_next_token_bitmask()
    rejected_token_ids = GrammarMatcher.get_rejected_tokens_from_bitmask(
        bitmask, matcher.mask_vocab_size
    )
    rejected_sizes.append(len(rejected_token_ids))
    if expected_rejected_sizes is not None:
        assert rejected_sizes[-1] == expected_rejected_sizes[-1]


if __name__ == "__main__":
    pytest.main(sys.argv)
