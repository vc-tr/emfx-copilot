"""The traded EM FX universe.

Each currency is quoted against USD in market convention (``local per USD``,
e.g. USDKRW = 1350 means 1350 KRW buys 1 USD). ``deliverable=False`` flags pairs
that are predominantly traded as **non-deliverable forwards (NDFs)** — the bread
and butter of an EM Asia / LatAm desk (KRW, TWD, INR, IDR, PHP, CLP, COP, ...).

Policy rates are *indicative* levels used to drive carry and covered-interest
forward pricing in the synthetic market. They are illustrative, not live quotes.
"""

from __future__ import annotations

from dataclasses import dataclass

# Indicative USD funding rate (think SOFR), annualised decimal.
USD_RATE: float = 0.0525


@dataclass(frozen=True)
class Currency:
    code: str  # ISO 4217
    name: str
    region: str  # "EM Asia" | "LatAm" | "EMEA" | "CEE"
    deliverable: bool  # False => typically traded NDF vs USD
    policy_rate: float  # indicative annual policy/depo rate (decimal)


UNIVERSE: tuple[Currency, ...] = (
    # EM Asia (NDF-heavy, plus a couple of deliverables)
    Currency("KRW", "South Korean won", "EM Asia", False, 0.0350),
    Currency("TWD", "Taiwan dollar", "EM Asia", False, 0.0200),
    Currency("INR", "Indian rupee", "EM Asia", False, 0.0650),
    Currency("IDR", "Indonesian rupiah", "EM Asia", False, 0.0600),
    Currency("PHP", "Philippine peso", "EM Asia", False, 0.0650),
    Currency("VND", "Vietnamese dong", "EM Asia", False, 0.0450),
    Currency("CNH", "Offshore Chinese yuan", "EM Asia", True, 0.0250),
    Currency("THB", "Thai baht", "EM Asia", True, 0.0250),
    # LatAm
    Currency("BRL", "Brazilian real", "LatAm", True, 0.1050),
    Currency("MXN", "Mexican peso", "LatAm", True, 0.1100),
    Currency("CLP", "Chilean peso", "LatAm", False, 0.0550),
    Currency("COP", "Colombian peso", "LatAm", False, 0.0975),
    # EMEA
    Currency("ZAR", "South African rand", "EMEA", True, 0.0825),
    Currency("TRY", "Turkish lira", "EMEA", True, 0.4500),
    # CEE
    Currency("PLN", "Polish zloty", "CEE", True, 0.0575),
    Currency("HUF", "Hungarian forint", "CEE", True, 0.0650),
)

CODES: tuple[str, ...] = tuple(c.code for c in UNIVERSE)
BY_CODE: dict[str, Currency] = {c.code: c for c in UNIVERSE}
REGIONS: tuple[str, ...] = tuple(dict.fromkeys(c.region for c in UNIVERSE))


def get_currency(code: str) -> Currency:
    try:
        return BY_CODE[code.upper()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown currency {code!r}. Known: {', '.join(CODES)}") from exc


def is_ndf(code: str) -> bool:
    """True if the pair is typically traded as a non-deliverable forward."""
    return not get_currency(code).deliverable
