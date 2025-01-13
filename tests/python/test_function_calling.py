import sys

import pytest

import xgrammar as xgr
from xgrammar.testing import _parse_message


def test_tag_based_simple():
    """Test simple tag-based format with single parameter."""
    input_str = '<function=get_weather>{"location": "SF"}</function>'
    result = _parse_message(input_str)
    assert len(result) == 1
    assert result[0] == ("get_weather", {"location": "SF"})


def test_tag_based_multiple_params():
    """Test tag-based format with multiple parameters."""
    input_str = '<function=search>{"query": "python", "limit": "10", "filter": "true"}</function>'
    result = _parse_message(input_str)
    assert len(result) == 1
    assert result[0] == ("search", {"query": "python", "limit": "10", "filter": "true"})


def test_json_format_simple():
    """Test simple JSON format."""
    input_str = '{"name": "get_time", "parameters": {"zone": "PST"}}'
    result = _parse_message(input_str)
    assert len(result) == 1
    assert result[0] == ("get_time", {"zone": "PST"})


def test_json_format_multiple_params():
    """Test JSON format with multiple parameters."""
    input_str = """{"name": "create_user",
                   "parameters": {
                       "username": "john_doe",
                       "age": "25",
                       "active": "true",
                       "email": "john@example.com"
                   }}"""
    result = _parse_message(input_str)
    assert len(result) == 1
    assert result[0] == (
        "create_user",
        {"username": "john_doe", "age": "25", "active": "true", "email": "john@example.com"},
    )


def test_multiple_functions():
    """Test parsing multiple function calls in the same input."""
    input_str = """
    <function=get_weather>{"city": "SF"}</function>
    <function=get_time>{"zone": "PST"}</function>
    """
    result = _parse_message(input_str)
    assert len(result) == 2
    assert result[0] == ("get_weather", {"city": "SF"})
    assert result[1] == ("get_time", {"zone": "PST"})


def test_mixed_formats():
    """Test mixing tag-based and JSON formats."""
    input_str = """
    <function=get_weather>{"city": "SF"}</function>
    {"name": "get_time", "parameters": {"zone": "PST"}}
    """
    result = _parse_message(input_str)
    assert len(result) == 2
    assert result[0] == ("get_weather", {"city": "SF"})
    assert result[1] == ("get_time", {"zone": "PST"})


def test_empty_parameters():
    """Test function calls with empty parameters."""
    input_str = "<function=ping>{}</function>"
    result = _parse_message(input_str)
    assert len(result) == 1
    assert result[0] == ("ping", {})


def test_special_values():
    """Test handling of special JSON values."""
    input_str = """
    <function=test_values>{
        "null_value": null,
        "bool_true": true,
        "bool_false": false,
        "number": 42,
        "float": 3.14
    }</function>
    """
    result = _parse_message(input_str)
    assert len(result) == 1
    assert result[0] == (
        "test_values",
        {
            "null_value": "null",
            "bool_true": "true",
            "bool_false": "false",
            "number": "42.000000",
            "float": "3.140000",
        },
    )


def test_invalid_json():
    """Test handling of invalid JSON in parameters."""
    input_str = "<function=test>{invalid json}</function>"
    with pytest.raises(Exception):
        _parse_message(input_str)


def test_invalid_json_ignore_error():
    """Test ignore_error parameter with invalid JSON."""
    input_str = """
    <function=valid>{"param": "value"}</function>
    <function=invalid>{invalid json}</function>
    """
    result = _parse_message(input_str, ignore_error=True)
    assert len(result) == 1
    assert result[0] == ("valid", {"param": "value"})


def test_malformed_tag():
    """Test handling of malformed tags."""
    input_str = '<function=test{"param": "value"}'  # Missing closing tag
    with pytest.raises(Exception):
        _parse_message(input_str)


def test_empty_input():
    """Test handling of empty input."""
    result = _parse_message("")
    assert len(result) == 0
    assert result == []


def test_whitespace_handling():
    """Test handling of various whitespace in input."""
    input_str = """
        <function=test>
            {
                "param": "value"
            }
        </function>
    """
    result = _parse_message(input_str)
    assert len(result) == 1
    assert result[0] == ("test", {"param": "value"})


@pytest.mark.parametrize(
    "input_str,expected",
    [
        ('<function=test>{"a": "1"}</function>', [("test", {"a": "1"})]),
        ('{"name": "test", "parameters": {"a": "1"}}', [("test", {"a": "1"})]),
    ],
)
def test_parametrized_formats(input_str, expected):
    """Test both formats with parameterization."""
    result = _parse_message(input_str)
    assert result == expected


if __name__ == "__main__":
    pytest.main(sys.argv)
