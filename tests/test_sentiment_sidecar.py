from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.core import sidecar

MODEL = Path("data/models/finbert")
pytestmark = pytest.mark.skipif(
    not sidecar.available() or not (MODEL / "model.safetensors").exists(),
    reason="FinBERT is not set up: uv run poe finbert-setup",
)
POSITIVE = "Stocks rallied and the British pound gained."
NEGATIVE = "Shares slide as analyst cuts price target"
NEUTRAL = "Company will hold its annual shareholder meeting on Tuesday"


def score(texts: list[str]) -> dict[str, Any]:
    params = {"model_dir": str(MODEL.resolve()), "batch": 64}
    outcome = sidecar.run_job("sentiment", params, {"data": pd.DataFrame({"text": texts})}, 600)
    assert outcome.status == "ok" and outcome.data is not None, outcome.error
    return outcome.data


def by_label(found: dict[str, Any]) -> list[dict[str, float]]:
    labels = list(found["labels"])
    return [dict(zip(labels, row, strict=True)) for row in found["probabilities"]]


@pytest.fixture(scope="module")
def found() -> dict[str, Any]:
    return score([POSITIVE, NEGATIVE, NEUTRAL])


def test_each_text_is_one_row_of_three_probabilities_that_add_up_to_one(
    found: dict[str, Any],
) -> None:
    rows = np.array(found["probabilities"])
    assert rows.shape == (3, 3) and rows.sum(axis=1) == pytest.approx(1.0, abs=1e-5)
    assert set(found["labels"]) == {"positive", "negative", "neutral"}


def test_a_clear_sentence_goes_the_way_a_reader_would_say(found: dict[str, Any]) -> None:
    winners = [max(row.items(), key=lambda item: item[1])[0] for row in by_label(found)]
    assert winners == ["positive", "negative", "neutral"]


def test_a_text_scores_alike_alone_and_in_a_batch(found: dict[str, Any]) -> None:
    alone = score([NEGATIVE])
    assert alone["probabilities"][0] == pytest.approx(found["probabilities"][1], abs=1e-3)


def test_the_job_reports_the_pinned_revision_and_runs_without_a_network(
    found: dict[str, Any],
) -> None:
    fetch = Path("finrl_sidecar/fetch_finbert.py").read_text()
    assert f'REVISION = "{found["revision"]}"' in fetch
