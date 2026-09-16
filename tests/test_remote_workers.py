"""Tests for the remote worker-count handling (scripts/remote.py).

The home server must decide how many CPUs to use: a client-provided low
``--workers`` value must not leave most cores idle.
"""

from scripts import remote


def test_inject_workers_forces_override(monkeypatch):
    monkeypatch.setattr(remote, "WORKERS_OVERRIDE", "8")
    monkeypatch.setattr(remote, "_WORKERS_CACHE", None)
    out = remote._inject_workers(
        "heatmap", ["--asset=X", "--workers=2", "--param_ranges={}"])
    assert out[-1] == "--workers=8"
    assert "--workers=2" not in out


def test_inject_workers_detects_nproc(monkeypatch):
    monkeypatch.setattr(remote, "WORKERS_OVERRIDE", None)
    monkeypatch.setattr(remote, "_WORKERS_CACHE", None)
    monkeypatch.setattr(remote, "ssh_out", lambda cmd: "12")
    out = remote._inject_workers("automator", ["--asset=X"])
    assert out[-1] == "--workers=12"


def test_inject_workers_handles_separate_arg_form(monkeypatch):
    monkeypatch.setattr(remote, "WORKERS_OVERRIDE", "6")
    monkeypatch.setattr(remote, "_WORKERS_CACHE", None)
    out = remote._inject_workers("heatmap", ["--asset=X", "--workers", "1"])
    assert out[-1] == "--workers=6"
    assert "--workers" not in out


def test_inject_workers_ignores_non_heavy_tools(monkeypatch):
    monkeypatch.setattr(remote, "WORKERS_OVERRIDE", "8")
    assert remote._inject_workers("pnl", ["--workers=2"]) == ["--workers=2"]
    assert remote._inject_workers("series", []) == []


def test_remote_workers_falls_back_when_nproc_unavailable(monkeypatch):
    monkeypatch.setattr(remote, "WORKERS_OVERRIDE", None)
    monkeypatch.setattr(remote, "_WORKERS_CACHE", None)
    monkeypatch.setattr(remote, "ssh_out", lambda cmd: "")
    assert remote._remote_workers() == 4
