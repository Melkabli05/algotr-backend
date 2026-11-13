import argparse
import json
import logging
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import truststore

from app.core import audit
from app.core.broker import (
    get_crypto_data_client,
    get_news_client,
    get_stock_data_client,
    get_trading_client,
)
from app.features.lifecycle.service import Lifecycle
from app.features.scanner import backfill, lists
from app.features.scanner.news import NewsFeed, NewsStore
from app.features.scanner.sentiment import Scorer
from app.features.scanner.service import ROOT, Scanner
from app.features.systematic import agent_lab, research
from app.features.systematic.catalog import STRATEGIES
from app.quant import history

logger = logging.getLogger(__name__)

RESULTS = Path("/results")


def backtest(name: str, results: Path) -> Path:
    """Backtest one catalog strategy and keep the result as a file of the results volume."""
    spec = STRATEGIES[name]
    found = research.research(get_stock_data_client(), get_crypto_data_client(), spec)
    results.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = results / f"{name}-{stamp}.json"
    path.write_text(found.model_dump_json(indent=2))
    logger.info("Backtest of %s written to %s", name, path)
    return path


def lifecycle() -> int:
    """The scheduled research run: record today's decisions and apply every lifecycle gate."""
    manager = Lifecycle(
        lambda spec: history.cached_prices(
            get_stock_data_client(), get_crypto_data_client(), spec.universe
        )
    )
    moves = manager.evaluate()
    for move in moves:
        logger.info(
            "%s moved %s to %s: %s", move.strategy, move.from_stage, move.to_stage, move.reason
        )
    return len(moves)


def scanner() -> Scanner:
    headlines = NewsStore(ROOT / "news.sqlite")
    return Scanner(
        get_trading_client(),
        get_stock_data_client(),
        news=NewsFeed(get_news_client(), headlines),
        sentiment=Scorer(headlines),
    )


def evaluate() -> dict[str, bool]:
    """Judge the agents and their ensemble by the gates, with the market's headline mood."""
    outcome = agent_lab.evaluate(
        get_stock_data_client(),
        get_crypto_data_client(),
        audit.AuditLog(audit.AUDIT_PATH),
        agent_lab.AiConfig(kronos=(agent_lab.KRONOS_DIR / "revision.json").exists()),
        mood=scanner(),
    )
    return {group.name: group.verdict.eligible for group in outcome.groups}


def news_backfill(since: date) -> dict[str, int]:
    """Fetch the Shariah-listed stocks' headlines back to `since`, then score them with FinBERT."""
    headlines = NewsStore(ROOT / "news.sqlite")
    feed = NewsFeed(get_news_client(), headlines)
    members = {s for snap in lists.current(lists.SOURCES, ROOT / "lists") for s in snap.symbols}
    symbols = sorted(members | {"SPUS", "HLAL"})
    today = datetime.now(UTC).date()
    added = backfill.backfill(feed, symbols, since, today, report=logger.info)
    return {"added": added, "scored": Scorer(headlines).score_new()}


def sentiment_uncertainty() -> dict[str, float | int | None]:
    """Measure the uncertainty of every headline scored before it was kept, then calibrate
    the gate on all of them."""
    scorer = Scorer(NewsStore(ROOT / "news.sqlite"))
    measured = 0
    while done := scorer.measure():
        measured += done
        logger.info(json.dumps({"measured": measured}))
    return {"measured": measured, "tau": scorer.store.gate()}


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    truststore.inject_into_ssl()
    STRATEGIES.refresh()
    parser = argparse.ArgumentParser(prog="python -m app.jobs")
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("backtest", help="Backtest one strategy into the results volume")
    one.add_argument("strategy", choices=sorted(STRATEGIES))
    one.add_argument("--results", type=Path, default=RESULTS)
    commands.add_parser("lifecycle", help="Apply the lifecycle's gates now")
    commands.add_parser("ai-evaluate", help="Judge the agents by the gates (about 80 minutes)")
    news = commands.add_parser("news-backfill", help="Fetch and score past headlines")
    news.add_argument("--since", type=date.fromisoformat, default=date(2020, 5, 1))
    commands.add_parser(
        "sentiment-uncertainty",
        help="Measure FinBERT's uncertainty on past headlines and calibrate the gate (~75 min)",
    )
    args = parser.parse_args(argv)
    if args.command == "backtest":
        logger.info(json.dumps({"result": str(backtest(args.strategy, args.results))}))
    elif args.command == "lifecycle":
        logger.info(json.dumps({"moves": lifecycle()}))
    elif args.command == "ai-evaluate":
        logger.info(json.dumps({"eligible": evaluate()}))
    elif args.command == "sentiment-uncertainty":
        logger.info(json.dumps(sentiment_uncertainty()))
    else:
        logger.info(json.dumps(news_backfill(args.since)))


if __name__ == "__main__":
    main()
