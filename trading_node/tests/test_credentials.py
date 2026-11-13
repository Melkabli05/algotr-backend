from types import SimpleNamespace

import pytest

from trading_node.credentials import ModeMismatchError, check_mode


def account(number: str) -> SimpleNamespace:
    return SimpleNamespace(account_number=number)


def test_paper_keys_on_a_paper_account_in_paper_mode_pass() -> None:
    check_mode("paper", True, "PKABC", account("PA3XYZ"))


def test_a_live_key_on_the_paper_host_is_refused() -> None:
    with pytest.raises(ModeMismatchError, match="live key"):
        check_mode("paper", True, "AKABC", account("PA3XYZ"))


def test_a_live_account_answering_in_paper_mode_is_refused() -> None:
    with pytest.raises(ModeMismatchError, match="live account"):
        check_mode("paper", True, "PKABC", account("123456789"))


def test_the_host_flag_must_agree_with_the_mode() -> None:
    with pytest.raises(ModeMismatchError, match="ALPACA_PAPER"):
        check_mode("live", True, "AKABC", account("123456789"))
