# Alpaca adapter for NautilusTrader

A pure-Python adapter on the NautilusTrader **2.0.0rc7** Python adapter interface
([docs](https://nautilustrader.io/docs/latest/developer_guide/python_adapters/)), built on the
official [`alpaca-py`](https://github.com/alpacahq/alpaca-py) SDK for all networking.
Equities and crypto; paper or live depending on `ALPACA_PAPER`. The daily plan job
([`trading_node/run_plan.py`](../run_plan.py)) is its only user.

| File | Role |
| --- | --- |
| `config.py` | `AlpacaDataClientConfig`, `AlpacaExecClientConfig` |
| `parsing.py` | pure translation between alpaca-py models and Nautilus types |
| `providers.py` | instrument loading (`Equity`, crypto `CurrencyPair`) |
| `data.py`, `execution.py` | the two clients |
| `factories.py` | client construction for `LiveNode` |
| `streams.py`, `sdk.py` | stream threads and typed SDK-call helpers |

## What works

- **Data:** quotes, trades and 1-minute bars (streamed); historical bars; instrument requests.
- **Execution:** market and limit orders (`day`, `gtc`, `ioc`, `fok`), cancel, cancel-all, modify
  (Alpaca replace), account state, order and fill events from the trade-update stream, and startup
  reconciliation (orders, one aggregate fill per order, positions).
- Verified live on Alpaca paper: submit, accepted, filled, position opened and closed, and
  resting-order cancel, with unit and real-engine tests in `tests/`.

## Not supported

Stop, trailing and bracket orders, order lists, options, margin accounts (the account is modelled
as `CASH`/`NETTING`, so shorting is not available), fractional-share positions (Nautilus equities
are whole shares; such positions, orders and fills are skipped with a warning), and historical
quote/trade requests through the data client.

## Things worth knowing

- **One data connection.** Alpaca's free plan allows one connection per market-data stream. A second
  process on the same stream gets `connection limit exceeded` (logged by alpaca-py, so Python
  logging must be enabled to see it).
- **Crypto fees** are charged in the asset received. Buy commissions are rounded *up* to the
  currency precision on every fill so Nautilus never believes it holds more than Alpaca, otherwise a
  full-position sell is rejected by a few nano-coins. Stocks are commission-free.
- **Cash accounts** cannot sell what they do not hold, so Nautilus's risk engine denies a
  crypto/stock sell without a position.
- **Fills** in reconciliation are one aggregate fill per order: alpaca-py exposes per-fill activity
  only in its Broker API.
- **rc7 specifics.** Subscription commands carry one `instrument_id` / `bar_type`; the newer docs
  describe lists (`instrument_ids`). `TradeUpdate.event` arrives as a plain string even though it is
  typed as an enum. Port these when moving past rc7.
