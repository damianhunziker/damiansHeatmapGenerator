"""Tests for the test.py CLI argument parser used by ``--api``."""


def test_coerce_booleans(cli):
    assert cli._coerce_value("k", "true") is True
    assert cli._coerce_value("k", "False") is False


def test_coerce_numbers(cli):
    value = cli._coerce_value("k", "10")
    assert value == 10 and isinstance(value, int)
    assert cli._coerce_value("k", "1.5") == 1.5
    assert cli._coerce_value("k", "-3") == -3


def test_coerce_json_objects_and_arrays(cli):
    assert cli._coerce_value("param_ranges", '{"a": 1}') == {"a": 1}
    assert cli._coerce_value("k", "[1, 2]") == [1, 2]


def test_coerce_list_keys_split_on_commas(cli):
    assert cli._coerce_value("pairs", "BTCUSDT,ETHUSDT") == ["BTCUSDT", "ETHUSDT"]
    assert cli._coerce_value("pairs", "BTCUSDT") == ["BTCUSDT"]
    assert cli._coerce_value("columns", "a, b") == ["a", "b"]


def test_parse_equals_space_and_flag(cli):
    parsed = cli.parse_cli_params(["--a=1", "--b", "2", "--flag"])
    assert parsed == {"a": 1, "b": 2, "flag": True}


def test_parse_ignores_positional_arguments(cli):
    assert cli.parse_cli_params(["foo", "--x=1"]) == {"x": 1}
