import io
import os
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd
import pytest

from app.quant import proxy

TEXT = """This file was created by using the 202609 CRSP database.
The Tbill return is the simple daily rate.

,Mkt-RF,SMB,HML,RF
19260701,    0.09,   -0.23,   -0.28,    0.01
19260702,    0.45,   -0.34,   -0.03,    0.01
20260930,   -0.26,    0.32,   -0.88,    0.01

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def archive(text: str = TEXT) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as found:
        found.writestr("F-F_Research_Data_Factors_daily.csv", text)
    return buffer.getvalue()


def client_serving(body: bytes, status: int = 200) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, content=body))
    )


def test_the_market_return_is_the_excess_return_plus_the_bill_rate() -> None:
    returns = proxy.parse(archive())
    assert returns.tolist() == pytest.approx([0.001, 0.0046, -0.0025])
    assert returns.index[0] == pd.Timestamp("1926-07-01")


def test_a_file_without_daily_rows_is_refused() -> None:
    with pytest.raises(proxy.ProxyFormatError):
        proxy.parse(archive("nothing here"))


def test_a_download_is_kept_and_read_back(tmp_path: Path) -> None:
    path = tmp_path / "us-market.csv"
    assert proxy.market(path) is None
    proxy.download(path, client_serving(archive()))
    stored = proxy.market(path)
    assert stored is not None
    assert stored.tolist() == pytest.approx([0.001, 0.0046, -0.0025])


def test_the_file_is_stale_when_missing_or_older_than_five_weeks(tmp_path: Path) -> None:
    path = tmp_path / "us-market.csv"
    assert proxy.stale(path)
    proxy.download(path, client_serving(archive()))
    assert not proxy.stale(path)
    old = (datetime.now(UTC) - timedelta(days=40)).timestamp()
    os.utime(path, (old, old))
    assert proxy.stale(path)


def test_a_failed_refresh_keeps_the_last_good_file(tmp_path: Path) -> None:
    path = tmp_path / "us-market.csv"
    proxy.download(path, client_serving(archive()))
    old = (datetime.now(UTC) - timedelta(days=40)).timestamp()
    os.utime(path, (old, old))
    assert proxy.refresh(path) is False
    assert proxy.market(path) is not None
