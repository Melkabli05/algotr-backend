import re
from typing import Literal, NamedTuple

from pydantic import BaseModel

Kind = Literal[
    "halt",
    "guidance",
    "legal",
    "offering",
    "regulatory",
    "mna",
    "management",
    "insider",
    "capital_return",
    "analyst",
    "results",
    "contract",
    "market",
    "other",
]
Direction = Literal["positive", "negative", "neutral"]
Severity = Literal["high", "medium", "low"]
MARKET_TAGS = 6


class Event(BaseModel):
    kind: Kind
    direction: Direction
    severity: Severity
    label: str

    @property
    def veto(self) -> bool:
        """Whether a new buy should wait: a serious setback, or a deal that pins the price."""
        return self.severity == "high" and (self.direction == "negative" or self.label == PENDING)


class Rule(NamedTuple):
    kind: Kind
    pattern: re.Pattern[str]
    direction: Direction
    severity: Severity
    label: str


PENDING = "Deal pending"


def rule(kind: Kind, pattern: str, direction: Direction, severity: Severity, label: str) -> Rule:
    return Rule(kind, re.compile(pattern, re.IGNORECASE), direction, severity, label)


RULES = (
    rule("other", r"\bonchain\b|\btokeni[sz]ed\b", "neutral", "low", "Tokenized stock"),
    rule("other", r"\binitial public offering\b|\bIPO\b", "neutral", "low", "IPO"),
    rule(
        "other",
        r"\b(interest|participat\w+|invests?|investing)\b[^.;]{0,60}\boffering\b",
        "neutral",
        "low",
        "Investor in an offering",
    ),
    rule("halt", r"\bquotation resumption\b|\btrading resum", "neutral", "low", "Trading resumes"),
    rule(
        "halt",
        r"\btrading halt\b|\bhalted\b|\bdelist\w*|\bbankruptcy\b|\bchapter 11\b"
        r"|\bsuspends? trading",
        "negative",
        "high",
        "Halt or delisting",
    ),
    rule(
        "guidance",
        r"\b(raises?|boosts?|lifts?|hikes?)\b[^;]{0,60}\b(guidance|outlook)\b.*"
        r"\b(lowers?|cuts?|trims?|reduces?)\b[^;]{0,60}\b(guidance|outlook)\b"
        r"|\b(lowers?|cuts?|trims?|reduces?)\b[^;]{0,60}\b(guidance|outlook)\b.*"
        r"\b(raises?|boosts?|lifts?|hikes?)\b[^;]{0,60}\b(guidance|outlook)\b",
        "neutral",
        "medium",
        "Guidance mixed",
    ),
    rule(
        "guidance",
        r"\b(cuts|cut|lowers|lowered|trims|trimmed|reduces|reduced|withdraws|withdrew|suspends"
        r"|suspended|slashes|slashed)\b[^.;]{0,40}\b(guidance|outlook)\b",
        "negative",
        "high",
        "Guidance cut",
    ),
    rule(
        "legal",
        r"\b(SEC|DOJ|FTC|FBI|department of justice)\b[^.;]{0,30}"
        r"\b(charges?|probe|investigat\w+|subpoena\w*|complaint|sues?)\b"
        r"|\bsubpoena\w*|\bindict\w*|\bcriminal (charges?|probe|investigation|complaint)\b",
        "negative",
        "high",
        "Regulator action",
    ),
    rule(
        "legal",
        r"\b(European Commission|competition authority|CMA|attorneys? general|SEC|DOJ|FTC)\b"
        r"[^.;]{0,40}\b(investigat\w+|probe|inquiry|charges?|fines?)\b",
        "negative",
        "medium",
        "Regulator probe",
    ),
    rule(
        "offering",
        r"\b(announces|prices|launches|files|proposes|commences|upsizes)\b[^.;]{0,40}"
        r"\b(public|secondary|common stock|equity) offering\b|\bat-the-market\b|\bATM program\b",
        "negative",
        "high",
        "Equity offering",
    ),
    rule(
        "regulatory",
        r"\brecall\w*|\bcomplete response letter\b|\bCRL\b|\bclinical hold\b|\bFDA rejects?\b",
        "negative",
        "high",
        "Recall or FDA setback",
    ),
    rule(
        "regulatory",
        r"\bwarning letter\b|\bsafety order\b|\bcease.and.desist\b|\bconsent (order|decree)\b",
        "negative",
        "medium",
        "Regulatory order",
    ),
    rule("analyst", r"\bdowngrad\w+", "negative", "medium", "Downgrade"),
    rule("analyst", r"\bupgrad\w+", "positive", "medium", "Upgrade"),
    rule(
        "analyst",
        r"\b(lowers?|cuts?|slashes?)\b[^.;]{0,20}\bprice targets?\b",
        "negative",
        "low",
        "Price target cut",
    ),
    rule(
        "analyst",
        r"\b(raises?|boosts?|lifts?|hikes?|increases?)\b[^.;]{0,20}\bprice targets?\b",
        "positive",
        "low",
        "Price target raised",
    ),
    rule(
        "analyst",
        r"\b(initiat\w+|resum\w+|reiterat\w+|reaffirm\w+|maintain\w+)\b[^.;]{0,40}"
        r"\b(coverage|rating|buy|hold|sell|neutral|outperform|underperform|overweight"
        r"|underweight|equal.?weight|market perform|sector (perform|weight)|peer perform)\b"
        r"|\bprice targets?\b",
        "neutral",
        "low",
        "Analyst note",
    ),
    rule(
        "guidance",
        r"\b(raises?|boosts?|increases?|lifts?|hikes?)\b[^.;]{0,40}\b(guidance|outlook|forecast)\b",
        "positive",
        "medium",
        "Guidance raised",
    ),
    rule(
        "guidance",
        r"\bguidance\b|\b(reaffirms?|reiterates?|updates?|narrows?)\b[^.;]{0,40}\boutlook\b",
        "neutral",
        "low",
        "Guidance",
    ),
    rule(
        "legal",
        r"\b(wins|dismiss\w*|clears?|cleared|approv\w+|upholds?)\b[^.;]{0,60}"
        r"\b(lawsuits?|antitrust|cases?|claims?)\b",
        "neutral",
        "low",
        "Legal ruling",
    ),
    rule(
        "legal",
        r"\bclass action\b|\blawsuits?\b|\bsues?\b|\bsued\b|\bantitrust\b|\bfined?\b|\bpenalty\b"
        r"|\bsecurities (fraud|laws?) violations?\b",
        "negative",
        "medium",
        "Legal action",
    ),
    rule(
        "legal",
        r"\binvestigation (into|of|on behalf)\b|\bannounces investigation\b",
        "negative",
        "low",
        "Law-firm notice",
    ),
    rule(
        "offering",
        r"\b(convertible|senior|debt) (notes )?offering\b|\bprospectus\b|\bshelf\b"
        r"|\bprivate placement\b|\bnotes due\b|\bprices? \$[\d.]+ ?[bm]\w* senior notes\b",
        "negative",
        "medium",
        "Debt or shelf filing",
    ),
    rule(
        "regulatory",
        r"\bFDA\b[^.;]{0,40}\b(approv\w+|clear\w+|grants?)\b"
        r"|\bapproves?\b[^.;]{0,60}\b(FDA|EMA|MHRA|regulator)\b|\bapproval\b",
        "positive",
        "medium",
        "Regulatory approval",
    ),
    rule(
        "regulatory",
        r"\b(FDA|EMA|clinical|trial|phase [123])\b",
        "neutral",
        "low",
        "Clinical or regulatory news",
    ),
    rule(
        "mna",
        r"\bto be acquired\b|\bagrees? to be (acquired|bought|taken private)\b|\btake-?private\b"
        r"|\btaken private\b|\bbuyout offer\b|\btakeover (bid|offer|approach)\b"
        r"|\breceives? (an? )?(unsolicited )?(offer|bid|proposal)\b",
        "neutral",
        "high",
        PENDING,
    ),
    rule(
        "mna",
        r"\bacquir(es|ed|ing)\b|\bacquisition\b|\bto acquire\b|\bmerger\b|\bmerge\b"
        r"|\bdivest\w*|\bspin-?off\b",
        "neutral",
        "medium",
        "Acquisition or merger",
    ),
    rule(
        "management",
        r"\b(CEO|CFO|COO|chair(man|woman)?|president|executive|leadership)\b[^.;]{0,50}"
        r"\b(steps? down|resign\w*|depart\w*|retire\w*|transition|succession|exits?|ousted"
        r"|fired)\b",
        "negative",
        "medium",
        "Leadership change",
    ),
    rule(
        "management",
        r"\b(appoints?|names|elects?)\b",
        "neutral",
        "low",
        "Appointment",
    ),
    rule(
        "insider",
        r"\b(sells?|sold|dumps?|trims?)\b[^;]{0,30}\b(shares|stake|stock|holdings?)\b"
        r"|\bshare sale\b",
        "negative",
        "low",
        "Insider or fund selling",
    ),
    rule(
        "insider",
        r"\b(insiders?|directors?|executives?|officers?|CEO|CFO|chair\w*|congress\w*)\b[^.;]{0,30}"
        r"\b(buys?|bought|purchases?)\b",
        "positive",
        "low",
        "Insider or fund buying",
    ),
    rule(
        "capital_return",
        r"\b(suspends?|cuts?|eliminates?)\b[^.;]{0,20}\bdividend\b",
        "negative",
        "high",
        "Dividend cut",
    ),
    rule(
        "capital_return",
        r"\b(raises?|increases?|boosts?|special)\b[^.;]{0,30}\b(dividend|buyback|repurchase)\b"
        r"|\bbuyback\b|\brepurchase\b",
        "positive",
        "low",
        "Dividend or buyback raised",
    ),
    rule("capital_return", r"\bdividend\b", "neutral", "low", "Dividend"),
    rule(
        "results",
        r"\b(beats?|tops?|exceeds?)\b[^;]{0,50}\b(estimates?|expectations?|consensus)\b",
        "positive",
        "medium",
        "Results beat",
    ),
    rule(
        "results",
        r"\b(misses?|falls? short of)\b[^;]{0,50}\b(estimates?|expectations?|consensus)\b",
        "negative",
        "medium",
        "Results miss",
    ),
    rule(
        "results",
        r"\b(Q[1-4]|first|second|third|fourth)[ -](quarter|fiscal)\b|\bfiscal (year )?\d{4}\b"
        r"|\b(earnings|EPS|financial results|results|conference call|preview|print)\b",
        "neutral",
        "low",
        "Results",
    ),
    rule(
        "other",
        r"\bstock\b[^.;]{0,25}\b(plunges?|tumbles?|falls?|drops?|sinks?|slides?|slumps?)\b"
        r"|\bshares? (fall|drop|slump|sink)\b",
        "negative",
        "low",
        "Share price falls",
    ),
    rule(
        "other",
        r"\bstock\b[^.;]{0,25}\b(surges?|jumps?|soars?|rallies|climbs?|spikes?)\b",
        "positive",
        "low",
        "Share price rises",
    ),
    rule("analyst", r"\banalysts?\b", "neutral", "low", "Analyst note"),
    rule(
        "contract",
        r"\b(contract|agreement|partnership|collaborat\w+|licen[cs]e|award\w*|selected|launch\w*"
        r"|unveil\w*|introduc\w*|announces?)\b",
        "neutral",
        "low",
        "Business news",
    ),
)
OTHER = Event(kind="other", direction="neutral", severity="low", label="Other")
COMMENTARY = Event(kind="market", direction="neutral", severity="low", label="Market commentary")


def classify(headline: str, tags: int = 1) -> Event:
    """The first rule that matches the headline; articles tagged to many stocks are commentary."""
    if tags >= MARKET_TAGS:
        return COMMENTARY
    for item in RULES:
        if item.pattern.search(headline):
            return Event(
                kind=item.kind,
                direction=item.direction,
                severity=item.severity,
                label=item.label,
            )
    return OTHER
