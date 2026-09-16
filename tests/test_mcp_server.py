"""Tests for the MCP server wiring (skipped if the 'mcp' package is missing)."""

import pytest

pytest.importorskip("mcp")

EXPECTED_TOOLS = {
    "get_schema", "fetch_data", "run_pnl", "run_chart_analysis",
    "run_heatmap", "run_automator", "get_series", "run_tool",
    "list_artifacts", "read_artifact",
}


def _registered_tool_names():
    import mcp_server.server as server

    manager = getattr(server.mcp, "_tool_manager", None)
    if manager is None:
        pytest.skip("tool manager not introspectable for this mcp version")
    return {tool.name for tool in manager.list_tools()}


def test_all_tools_registered():
    assert EXPECTED_TOOLS <= _registered_tool_names()


def test_server_uses_project_root_on_path():
    import mcp_server.server as server

    assert server.TEST_PY.endswith("test.py")
    assert server.PROJECT_ROOT


@pytest.mark.integration
def test_call_schema_bridge():
    import mcp_server.server as server

    envelope = server._call("schema", {}, timeout=300)
    assert envelope["ok"] is True
    assert "pnl" in envelope["data"]["tools"]
