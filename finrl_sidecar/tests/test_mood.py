import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import runner


def bars(days: int = 120, mood: bool = True) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=days).strftime("%Y-%m-%d")
    rng = np.random.default_rng(4)
    rows = []
    for tic, base in (("HLAL", 40.0), ("SPUS", 30.0)):
        closes = base * np.exp(np.cumsum(rng.normal(0, 0.01, days)))
        for index, day in enumerate(dates):
            row = {
                "date": day,
                "tic": tic,
                "open": closes[index],
                "high": closes[index] * 1.01,
                "low": closes[index] * 0.99,
                "close": closes[index],
                "volume": 1e6,
            }
            if mood:
                row["sentiment"] = 0.4 if index == days - 1 else np.nan
                row["news_risk"] = 5.0 if index == days - 1 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


class MoodFeatures(unittest.TestCase):
    def prepared(self, frame: pd.DataFrame) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "data.csv")
            frame.to_csv(path, index=False)
            features, _ = runner.prepare_agents({"files": {"data": str(path)}})
        return features

    def test_the_mood_is_one_market_feature_each_and_a_quiet_day_reads_neutral(self) -> None:
        features = self.prepared(bars())
        self.assertIn("sentiment", features.columns)
        self.assertIn("news_risk", features.columns)
        self.assertAlmostEqual(features["sentiment"].iloc[-1], 0.4)
        self.assertAlmostEqual(features["news_risk"].iloc[-1], 1.0)
        self.assertTrue((features["sentiment"].iloc[:-1] == 0).all())

    def test_a_later_mood_is_never_filled_back_into_earlier_days(self) -> None:
        features = self.prepared(bars())
        self.assertEqual(int((features["sentiment"] != 0).sum()), 1)

    def test_without_mood_columns_the_features_are_as_before(self) -> None:
        features = self.prepared(bars(mood=False))
        self.assertNotIn("sentiment", features.columns)
        self.assertTrue(all(":" in column for column in features.columns))


if __name__ == "__main__":
    unittest.main()
