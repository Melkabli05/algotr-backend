# Headline sentiment

Every headline the scanner stores gets a sentiment record from FinBERT, run in the FinRL sidecar. The record is shown beside the headline and kept for later. Nothing trades on it, ranks by it or vetoes by it: the veto still comes from the rules ([scanner.md](scanner.md)). What the feature list claimed, what was measured and what is not built: [sentiment-research.md](sentiment-research.md).

| Endpoint | What it shows |
| --- | --- |
| `GET /scanner/news/{symbol}` | each headline with its `sentiment` record, or `null` while it waits |
| `GET /scanner/status` | `news.sentiment`: whether scoring is on, the model, how many records there are, how many headlines wait, the share under 80% confidence, when the last was scored and the last error |

There is no setting. Scoring starts by itself once the sidecar and the weights are installed:

```bash
uv run poe finbert-setup
```

That installs the sidecar if it is missing and downloads the five files of [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) at commit `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031` (438 MB) into `data/models/finbert/`. Each file is checked against a SHA-256 pinned in `finrl_sidecar/fetch_finbert.py` and deleted when it differs. The sidecar then loads the model offline, and the scorer refuses a result from any other revision. The weights are licensed CC BY-NC-SA 4.0, non-commercial; see [the research](sentiment-research.md#the-licence).

## The record

| Field | What it holds |
| --- | --- |
| `score` | the probability of positive less the probability of negative, from -1.0 to +1.0 |
| `confidence` | the highest of the three probabilities; with three classes it cannot be under 0.33 |
| `positive`, `negative`, `neutral` | the three probabilities, which sum to 1 |
| `risk` | 1 to 5, read from the rule typing of the headline (below); `risk_basis` says `rules` |
| `model` | `ProsusAI/finbert@0bc2ac1`: the repository and the first seven characters of the commit |
| `processed` | when the text was first scored, in UTC |
| `text_hash` | SHA-256 of the headline with surrounding whitespace removed |
| `fallback` | true when `confidence` is under 0.8 |

FinBERT has three outputs and no risk. The risk score comes from the rules that already type every headline (`features/scanner/headlines.py`), so it is as good as they are, and it can disagree with the score: a strongly negative note on an analyst's valuation view has a risk of 1.

| Risk | When |
| --- | --- |
| 5 | the rules call it a veto: a high-severity negative event, or a pending deal |
| 4 | a negative event of medium severity: a downgrade, a probe, a lawsuit, a leadership change |
| 3 | a negative event of low severity: a price target cut, insider selling |
| 2 | a neutral event of medium severity: mixed guidance, an acquisition |
| 1 | everything else |

## How headlines are scored

After every news poll (every 15 minutes and once at startup) the scorer takes the stored headlines that have no record, newest first, 2,000 at a time. A chunk is hashed. Texts that already have a record from this model are not sent anywhere; the rest go to one sidecar job (the task `sentiment`, 64 texts a batch, on the Mac's GPU when there is one). The records are inserted and every article in the chunk is linked to the record of its text. The first run scored the 11,215 headlines then stored in under a minute; a poll that finds two new ones takes about 10 seconds, most of it loading the model, which every job does again.

One run goes at a time, and a second caller returns at once. A job that fails leaves its chunk unlinked, sets `error` in the status and is tried again at the next poll. Without the sidecar or the weights the status says scoring is off and everything else in the scanner works as before.

## The cache

The records are in `data/scanner/news.sqlite`, beside the headlines:

| Table | What it holds |
| --- | --- |
| `sentiment_cache` | one record for each text and model, keyed on `(text_hash, model)`; written once and never changed, so `processed` is when the text was first scored |
| `article_sentiment` | which text each article has |

A text is scored once. The same story under several symbols is one article with several tags, and a feed's repeat of a note shares the record of the first (299 of 11,215 headlines in the first run), so a duplicate costs a lookup. The key holds the model, so a second model can keep its records beside these without a migration. The list asked for PostgreSQL and a Redis cache with a 5 minute life; neither is in the stack, and a record that expires after 5 minutes would be rescored later and would not be there to audit.

## The fallback flag

`fallback` marks a headline whose confidence is under 0.8, which is 42% of those stored. The list sends those to a next stage that it did not describe and that is not built, so nothing reads the flag. At 0.8 it also does not separate good answers from bad ones ([the measurements](sentiment-research.md#how-sure-is-it)).

## On the dashboard

The PICKS view says how many headlines are scored, how many wait and what share is under the 80% line, or how to turn scoring on; a failure is a warning. The headlines of the selected stock gain Sentiment, Confidence and Risk columns, which are empty for a headline that waits and are left out when none of them has a record.

## Where the code is

| Module | What it holds |
| --- | --- |
| `features/scanner/sentiment.py` | the record, the risk mapping, the store and the scorer |
| `features/scanner/news.py` | the `Sentiment` model carried by a `Headline` |
| `features/scanner/watch.py` | scoring after each poll, and the records on the headlines it reports |
| `core/sidecar.py` | the bridge to the sidecar, shared with the agents |
| `finrl_sidecar/runner.py` | the task `sentiment` |
| `finrl_sidecar/fetch_finbert.py` | the pinned, hash-checked download (`uv run poe finbert-setup`) |
