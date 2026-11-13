# Headline sentiment: research

The sixth feature list (5.1, FinBERT as the primary classifier, and 5.3, the output format; 5.2, the stage it falls back to, was not in the list) read against the model's paper and licence, 491 hand-scored financial headlines it was not trained on, and the 11,215 headlines this scanner stored in the 60 days to 3 November 2026, with the scanner's price panel. All figures were measured on 3 November 2026 with the pinned weights, on the Mac's GPU. Earlier work: [scanner-research.md](scanner-research.md), [scanner-features-research.md](scanner-features-research.md), [ml-research.md](ml-research.md).

## Claims in the list that the evidence does not support

| Claim | What the evidence says |
| --- | --- |
| 97% accuracy on full-agreement sentences and 86% overall | These are the paper's figures on the Financial PhraseBank: 4,845 sentences from company news, 59.4% of them neutral, labelled by 16 annotators; accuracy 0.86 and F1 0.84 on all of it, 0.97 and 0.95 on the part where every annotator agreed ([Araci, 2019](https://arxiv.org/abs/1908.10063)). They are not headline figures, the released model was fine-tuned on that set so the set cannot check it again, and the model card reports none. On 491 headlines from another set (FiQA 2018) its three-class accuracy is 63%, against 55% for always answering "positive" |
| Fast enough to run locally on every incoming item without API cost | True, at a price in setup: 0.1 billion parameters, a 438 MB download, and torch, which cannot sit beside this project's dependencies. It runs in the FinRL sidecar. The 11,215 stored headlines took 31 to 50 seconds on the Mac's GPU, and every job loads the model again, 5 to 12 seconds |
| Falls back to the next stage when its confidence is under 0.8 | 42% of the stored headlines are under 0.8 (41.8%), and the line does not separate right from wrong: on FiQA headlines 65% of the answers at 0.8 or more are right against 57% below, mean confidence is 0.83 against accuracy 0.63 and the answers between 0.9 and 0.95 are right 60% of the time (below) |
| The record holds a risk score from 1 to 5 | FinBERT has three outputs, positive, negative and neutral, and no risk. The risk score is read from the scanner's rule typing, labelled `risk_basis: rules`, so it inherits their misses (below) |
| Records are stored in PostgreSQL (`sentiment_cache`) | PostgreSQL is not in the stack. `sentiment_cache` is a SQLite table in `news.sqlite` with the same fields |
| Records are cached in Redis for 5 minutes to avoid reprocessing duplicates | Redis is not in the stack, and an expiry is the wrong tool: a text is scored once and its record kept, so a duplicate costs a lookup now and next month, and the record is there for a backtest or an audit. In the first run 299 of 11,215 headlines repeated the text of another |
| It is the primary classifier for all headlines and short articles | Headlines only. The store keeps the headline and a summary, never the article, and only the headline is scored |
| (implied) A sentiment score is a signal | It describes the wording and the same day's move and does not predict the next one (below) |

## The licence

The weights are CC BY-NC-SA 4.0: no commercial use, and anything built from them is shared alike. That is fine for research on a paper account. Whether trading real money on a personal account is commercial use is not something this repository can settle; it is the licence holder's call or a lawyer's, and it should be settled before any live account depends on the score. Swapping the model means changing the pins in `finrl_sidecar/fetch_finbert.py` and the repository and revision in `sentiment.py`.

## How sure is it

FiQA 2018 task 1 ([pauri32/fiqa-2018](https://huggingface.co/datasets/pauri32/fiqa-2018)): 1,213 financial texts scored by hand from -1 to +1, of which 491 are headlines.

| Measure on the 491 headlines | Result |
| --- | --- |
| Spearman correlation of FinBERT's score with the human score | +0.75 (posts, which are not headlines: +0.51) |
| Where the human score is clear (345 headlines, over 0.2 either way): the sign of FinBERT's score agrees | 92% |
| The same, its most likely class agrees | 68%, because it answers "neutral" on 27% of them |
| Three-class accuracy | 63%; always "positive" scores 55% (positive 55%, negative 29%, neutral 16%) |
| The commonest error | a headline the annotators found positive that it calls neutral (117) |

FiQA calls anything above +0.1 positive, a low bar, so part of the gap may be where each set draws the line between positive and neutral and not a fault in the model. Either way, a number from the PhraseBank does not carry over to headlines.

Is a confidence of 0.8 meaningful? The probabilities are not calibrated:

| Confidence | Headlines | Mean confidence | Accuracy |
| --- | --- | --- | --- |
| up to 0.5 | 12 | 0.46 | 58% |
| 0.5 to 0.6 | 41 | 0.55 | 68% |
| 0.6 to 0.7 | 36 | 0.66 | 50% |
| 0.7 to 0.8 | 54 | 0.75 | 54% |
| 0.8 to 0.9 | 123 | 0.86 | 55% |
| 0.9 to 0.95 | 160 | 0.93 | 60% |
| over 0.95 | 65 | 0.96 | 94% |

At 0.8 or more 348 headlines are 65% right and below it 143 are 57% right. The expected calibration error is 0.24. Only above 0.95 does confidence mean what it says, and that is 13% of the headlines.

## On the scanner's headlines

The 11,215 stored headlines come out 49.6% neutral, 34.8% positive and 15.6% negative, with a mean score of +0.17. The quartiles of the confidence are 0.69, 0.84 and 0.91, and 41.8% are under 0.8 (Benzinga 42.7%, fiscal.ai 38.8%). Analyst notes are the least sure, 54% under the line: "Deutsche Bank Adjusts Ralph Lauren Price Target to $452 From $446" scores 0.34 for its most likely class.

Against the rules that type every headline:

| | Rules | FinBERT |
| --- | --- | --- |
| Headlines on which they commit to a side | 2,857 of 11,215 (25%) | all of them |
| Agreement where the rules commit | | 66% agree, 24% say neutral, 9% say the opposite |
| The 42 high-severity negative headlines (the veto kind) | all, by definition | 74% called negative, mean score -0.59 |
| The 28 clear FiQA headlines where the rules commit | right 96% | right 86% |

The rules say little and are exact; FinBERT covers everything and is looser. It would not replace the veto, since a quarter of the headlines the rules call a serious setback it does not call negative.

It scores the tone of a sentence and not what it means for the stock. "Nate Silver Says Democrats Are Clear Senate Favorites: 'A Really Bad N..." is -0.95 at 0.96 confidence, and it is tagged to Nvidia; "Argenx, Caribou Biosciences And Other Big Stocks Moving Lower In Thursday's Pre-Market Session" is -0.96, a report of a price move.

The risk score inherits the rules' misses. "Nvidia Stock Climbs Amid Oil Price Drop" is typed "Share price falls" (risk 3) while FinBERT scores it +0.06. A strongly negative sentiment can carry a risk of 1: "Apple Stock at 'Extreme Overvaluation': Analysts See Slowing Revenue G..." is -0.96, risk 1, an analyst note.

## What the score predicts

10,010 company-specific headlines (tagged to three stocks or fewer, one for each stock) in the 61 trading days to 3 November. The return is the stock's less the average of the followed names that day. The tradable entry is the next open after the headline (that day's for one before the open, the next trading day's for any other); the clocks assume US daylight time throughout, which misplaces an hour on the last two days after the change on 1 November. The interval resamples days, because the headlines of one day move together.

| Return | Headlines | Spearman with the score [95% interval] | Top fifth less bottom fifth |
| --- | --- | --- | --- |
| The same day, close to close (the reaction) | 10,010 | +0.094 [+0.062, +0.126] | +127.8 bps |
| The first tradable day, open to close | 10,010 | +0.008 [-0.017, +0.033] | +2.5 bps |
| The day after that | 9,464 | -0.008 [-0.034, +0.022] | -1.4 bps |

The same-day link is real and mostly mechanical: a headline that says a stock "climbs" or "tumbles" describes that day's move. The tradable returns show nothing, above 0.8 confidence (+0.011 [-0.023, +0.044], 5,809 headlines) or below it (+0.000 [-0.034, +0.037], 4,201). By band the mean same-day return runs from -133.7 bps for very negative scores to +27.2 bps for very positive ones, and the first tradable day from -10.6 to +7.6 bps with no order to it. Sixty-one days of one market is short: the interval says no rank correlation above about 0.03, not that none exists. Nothing here is a reason to rank by it or to trade on it.

## Decisions

- **FinBERT in the sidecar.** torch and transformers cannot live in the project's environment, and the sidecar already holds torch. The weights are pinned by commit and checked by hash, loaded with the hub switched off, and a result from any other revision is refused.
- **Libraries.** `transformers` 5.22.1 and `huggingface_hub` 2.4.0 in the sidecar's requirements, `hashlib.file_digest` for the check, SQLite for the store. The custom parts are the record, the store and the scorer, because no library keeps a record for each text and model, and the risk mapping, which reads rules the scanner already has. No classifier code is written.
- **One record for each text and model,** keyed on a hash, written once. A second model, or a second stage, adds records and replaces none.
- **The risk score is derived and says so.** It was the only way to give the field a value without a second model.
- **The 0.8 line is kept as a flag, taken from the list with its measured weakness stated here.** Nothing reads it.
- **Nothing trades, ranks or vetoes by the score.** The measurement above is why.

## Not built

The stage after 0.8 (5.2 was not sent), a persistent scoring worker that would load the model once, PostgreSQL, Redis, scoring of article bodies, sentiment as an input to the agents (60 days of scores is too short to train on) and any strategy or ranking on the score.

## Not verified live

The failure path (a sidecar that is missing or dies mid-chunk) was tested with stand-ins and not on the running stack. Scoring while a FinRL job holds the sidecar's cores was not tried. The real model is exercised by a test that is skipped when the weights are not installed. What was run live: the first scoring of the real store, a poll on the restarted stack that found two new headlines and had them scored and shown within 10 seconds, and the Scanner page with the new columns.
