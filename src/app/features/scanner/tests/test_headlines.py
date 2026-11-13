import pytest

from app.features.scanner import headlines
from app.features.scanner.headlines import PENDING, classify

CASES = [
    (
        "HC Wainwright & Co. Maintains Buy on Biogen, Raises Price Target to $270",
        "analyst",
        "positive",
        "low",
    ),
    (
        "Truist Securities Reiterates Hold on Stryker, Lowers Price Target to $305",
        "analyst",
        "negative",
        "low",
    ),
    ("Citigroup Reaffirms Neutral Rating on CRH plc", "analyst", "neutral", "low"),
    (
        "Guggenheim and Cantor Fitzgerald Adjust Price Targets on AbbVie",
        "analyst",
        "neutral",
        "low",
    ),
    ("Morgan Stanley Downgrades Hasbro to Equal-Weight", "analyst", "negative", "medium"),
    ("Bernstein Upgrades Nike to Outperform", "analyst", "positive", "medium"),
    (
        "Estee Lauder Cos Raises FY2027 GAAP EPS Guidance from $2.52-$2.85 to $2.85-$3.18",
        "guidance",
        "positive",
        "medium",
    ),
    (
        "First Solar Reports Q3 2026 Results and Lowers Full-Year Guidance",
        "guidance",
        "negative",
        "high",
    ),
    ("Peloton Withdraws Full-Year Outlook", "guidance", "negative", "high"),
    (
        "Reddit Reports Q3 2026 Earnings and Issues Q4 Revenue Guidance",
        "guidance",
        "neutral",
        "low",
    ),
    (
        "Trading Halt: Halted at 6:57:18 a.m. ET - Trading Halt: Halt News Pending",
        "halt",
        "negative",
        "high",
    ),
    (
        "Trading Halt: Halt status updated at 7:25:00 AM ET: Quotation Resumption: News Pending",
        "halt",
        "neutral",
        "low",
    ),
    (
        "Oracle Announces $10B Public Offering Of Common Stock; Also $10B Senior Notes Offering",
        "offering",
        "negative",
        "high",
    ),
    (
        "Carrier Global Files Prospectus For Offering Mixed Shelf; Terms Undisclosed",
        "offering",
        "negative",
        "medium",
    ),
    (
        "Securitize Expands Onchain U.S. Stock Offering to Include Arista Networks",
        "other",
        "neutral",
        "low",
    ),
    ("SEC Subpoenas Acme Corp Over Revenue Recognition", "legal", "negative", "high"),
    (
        "Investigation into Xylem Inc. Regarding Securities Law Violations",
        "legal",
        "negative",
        "medium",
    ),
    (
        "Law Firm Announces Investigation into Fortinet, Inc. Regarding Securities Laws",
        "legal",
        "negative",
        "low",
    ),
    ("Acme Faces Class Action Over Product Claims", "legal", "negative", "medium"),
    ("Medtronic Announces Voluntary Recall of Insulin Pumps", "regulatory", "negative", "high"),
    (
        "FDA Grants Tentative Approval for Generic Lenacapavir for HIV Prevention",
        "regulatory",
        "positive",
        "medium",
    ),
    (
        "Gilead Sciences to Present Clinical Data at HIV Glasgow 2026",
        "regulatory",
        "neutral",
        "low",
    ),
    (
        "Abbott's TECTONIC Trial Of Investigational SonicForce System Meets Primary Safety",
        "regulatory",
        "neutral",
        "low",
    ),
    ("Acme Agrees to Be Acquired by Contoso for $48 a Share", "mna", "neutral", "high"),
    (
        "Marvell Technology to Acquire Lumentum in $115 Billion All-Stock Merger",
        "mna",
        "neutral",
        "medium",
    ),
    ("NetApp Completes Acquisition of PEAK:AIO", "mna", "neutral", "medium"),
    (
        "TE Connectivity Announces Executive Leadership Transition",
        "management",
        "negative",
        "medium",
    ),
    ("Jabil Executive Vice President Matthew Crowley Sells Shares", "insider", "negative", "low"),
    ("SLB Insider Executes Share Sale", "insider", "negative", "low"),
    ("Ark Invest Trims Eli Lilly Holdings", "insider", "negative", "low"),
    (
        "Rockwell Automation Raises Quarterly Dividend From $1.38 To $1.45/Share",
        "capital_return",
        "positive",
        "low",
    ),
    ("Illinois Tool Works Inc. Declares Quarterly Dividend", "capital_return", "neutral", "low"),
    ("Acme Suspends Quarterly Dividend", "capital_return", "negative", "high"),
    (
        "Exxon Mobil Q3 Adj. EPS $4.23 Beats $4.08 Estimate, Sales $113.164B Beat Estimate",
        "results",
        "positive",
        "medium",
    ),
    ("Acme Q2 EPS $0.40 Misses $0.55 Estimate", "results", "negative", "medium"),
    ("Apple Scheduled to Report Fiscal Fourth-Quarter Earnings", "results", "neutral", "low"),
    (
        "AbbVie Enters Global License Agreement with Brise Pharmaceuticals for Migraine Therapies",
        "contract",
        "neutral",
        "low",
    ),
    (
        "Bernie Sanders Targets Elon Musk Over $1.6 Billion in Spending: ‘Trying To Buy...’",
        "other",
        "neutral",
        "low",
    ),
    (
        "Citigroup Maintains Buy on CrowdStrike Holdings, Raises Price Target to $295",
        "analyst",
        "positive",
        "low",
    ),
    (
        "Deckers Outdoor Raises FY2027 GAAP EPS Guidance from $7.35-$7.50 to $7.80-$7.95; "
        "Lowers FY2027 Sales Guidance",
        "guidance",
        "neutral",
        "medium",
    ),
    (
        "Mondelez International Reaffirms 2026 Organic Net Revenue Growth Guidance of At Least 2%",
        "guidance",
        "neutral",
        "low",
    ),
    (
        "FTC Initiates Inquiry into Data Usage by DoorDash and Other Delivery Platforms",
        "legal",
        "negative",
        "medium",
    ),
    (
        "Vulcan Materials Receives Regulatory Safety Order for California Quarry",
        "regulatory",
        "negative",
        "medium",
    ),
    ("CrowdStrike CEO George Kurtz Sells $5.3 Million in Shares", "insider", "negative", "low"),
    (
        "SanDisk Participates in Tenstorrent's $1.5 Billion Series E Funding Round",
        "other",
        "neutral",
        "low",
    ),
    (
        "Kimberly-Clark Prices $4.9 Billion Debt Offering to Fund Kenvue Acquisition",
        "offering",
        "negative",
        "medium",
    ),
]


@pytest.mark.parametrize(("headline", "kind", "direction", "severity"), CASES)
def test_a_headline_is_typed_with_its_direction_and_severity(
    headline: str, kind: str, direction: str, severity: str
) -> None:
    event = classify(headline)
    assert (event.kind, event.direction, event.severity) == (kind, direction, severity)


def test_an_article_tagged_to_many_stocks_is_commentary_whatever_it_says() -> None:
    event = classify("SEC Subpoenas Acme Corp Over Revenue Recognition", tags=headlines.MARKET_TAGS)
    assert event.kind == "market" and not event.veto


@pytest.mark.parametrize(
    "headline",
    [
        "First Solar Reports Q3 2026 Results and Lowers Full-Year Guidance",
        "Trading Halt: Halted at 6:57:18 a.m. ET - Trading Halt: Halt News Pending",
        "Oracle Announces $10B Public Offering Of Common Stock",
        "SEC Subpoenas Acme Corp Over Revenue Recognition",
        "Medtronic Announces Voluntary Recall of Insulin Pumps",
        "Acme Agrees to Be Acquired by Contoso for $48 a Share",
    ],
)
def test_a_serious_setback_or_a_pending_deal_is_a_veto(headline: str) -> None:
    assert classify(headline).veto


@pytest.mark.parametrize(
    "headline",
    [
        "Morgan Stanley Downgrades Hasbro to Equal-Weight",
        "Investigation into Xylem Inc. Regarding Securities Law Violations",
        "Marvell Technology to Acquire Lumentum in $115 Billion All-Stock Merger",
        "Acme Q2 EPS $0.40 Misses $0.55 Estimate",
        "TE Connectivity Announces Executive Leadership Transition",
        "Bank of America Lowers Fiscal 2027 iPhone Forecast for Apple",
        "CFO Says Capital Expenditures Were Lower Than Guidance",
        "Deckers Outdoor Raises FY2027 GAAP EPS Guidance from $7.35-$7.50 to $7.80-$7.95; "
        "Lowers FY2027 Sales Guidance",
        "FTC Initiates Inquiry into Data Usage by DoorDash and Other Delivery Platforms",
    ],
)
def test_a_bad_but_ordinary_headline_is_not_a_veto(headline: str) -> None:
    assert not classify(headline).veto


def test_the_pending_label_is_what_makes_a_neutral_deal_a_veto() -> None:
    event = classify("Acme Agrees to Be Acquired by Contoso for $48 a Share")
    assert (event.direction, event.label) == ("neutral", PENDING)
