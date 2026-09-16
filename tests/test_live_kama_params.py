"""Tests that LiveKAMASSLStrategy exposes scalable KAMA lengths."""

from classes.strategies.live_kama_ssl_strategy import LiveKAMASSLStrategy


def test_defaults_preserve_original_lengths():
    strategy = LiveKAMASSLStrategy()
    assert (strategy.entry_kama_length, strategy.entry_kama_fast, strategy.entry_kama_slow) == (16, 4, 24)
    assert (strategy.kama2_length, strategy.kama2_fast, strategy.kama2_slow) == (15, 3, 22)
    assert (strategy.exit_kama_length, strategy.exit_kama_fast, strategy.exit_kama_slow) == (14, 2, 20)


def test_scaled_lengths_are_applied():
    strategy = LiveKAMASSLStrategy(
        entry_kama_length=96, entry_kama_fast=29, entry_kama_slow=149,
        kama2_length=90, kama2_fast=20, kama2_slow=137,
        exit_kama_length=84, exit_kama_fast=17, exit_kama_slow=125,
    )
    assert (strategy.entry_kama_length, strategy.entry_kama_fast, strategy.entry_kama_slow) == (96, 29, 149)
    assert (strategy.kama2_length, strategy.kama2_fast, strategy.kama2_slow) == (90, 20, 137)
    assert (strategy.exit_kama_length, strategy.exit_kama_fast, strategy.exit_kama_slow) == (84, 17, 125)


def test_lengths_are_clamped_to_valid_minimums():
    strategy = LiveKAMASSLStrategy(
        entry_kama_length=0, entry_kama_fast=0, entry_kama_slow=1,
    )
    assert strategy.entry_kama_length >= 1
    assert strategy.entry_kama_fast >= 2
    assert strategy.entry_kama_slow >= 3
