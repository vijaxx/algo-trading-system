"""CLI argument validation -- bad input should print a clean message and a
non-zero exit code, not an uncaught traceback."""

from __future__ import annotations

from algotrade.cli import main


def test_backtest_rejects_unknown_strategy(capsys):
    code = main(["backtest", "--strategy", "nope", "--days", "2"])
    assert code == 2
    err = capsys.readouterr().err
    assert "unknown strategy" in err
    assert "nope" in err


def test_backtest_rejects_unknown_symbol(capsys):
    code = main(["backtest", "--strategy", "orb", "--symbols", "FAKESYM", "--days", "2"])
    assert code == 2
    err = capsys.readouterr().err
    assert "unknown symbol" in err
    assert "FAKESYM" in err
    # the valid universe should still be listed so the user knows what to use
    assert "RELIANCE" in err


def test_backtest_rejects_one_bad_symbol_among_valid_ones(capsys):
    code = main(
        [
            "backtest",
            "--strategy",
            "orb",
            "--symbols",
            "RELIANCE",
            "FAKESYM",
            "--days",
            "2",
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "FAKESYM" in err
    assert "RELIANCE" not in err.split(";")[0]  # only the bad symbol is flagged
