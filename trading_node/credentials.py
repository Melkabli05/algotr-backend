from typing import Any, Literal

Mode = Literal["paper", "live"]


class ModeMismatchError(RuntimeError):
    """The keys or the account belong to the other kind of account than the mode says."""


def account_kind(account_number: str) -> Mode:
    """Alpaca numbers paper accounts PA...; anything else is a live brokerage account."""
    return "paper" if account_number.upper().startswith("PA") else "live"


def key_kind(api_key: str) -> Mode:
    """Paper keys start PK; live keys start AK."""
    return "paper" if api_key.upper().startswith("PK") else "live"


def check_mode(mode: Mode, paper_flag: bool, api_key: str, account: Any) -> None:
    """Refuse to trade unless the mode, the host flag, the keys and the account all agree.

    A live key sent to the paper host is the dangerous direction: it answers, so only the
    account itself can tell which money is behind it.
    """
    problems = []
    if paper_flag != (mode == "paper"):
        problems.append(f"ALPACA_PAPER={paper_flag} disagrees with ALPACA_TRADING_MODE={mode}")
    if key_kind(api_key) != mode:
        problems.append(f"the API key is a {key_kind(api_key)} key")
    number = str(getattr(account, "account_number", "") or "")
    if account_kind(number) != mode:
        problems.append(f"account {number or '?'} is a {account_kind(number)} account")
    if problems:
        raise ModeMismatchError(f"Refusing to start in {mode} mode: " + "; ".join(problems))
