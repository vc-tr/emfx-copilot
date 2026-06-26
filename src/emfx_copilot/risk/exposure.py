"""Position book, exposure aggregation, and limit monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..data.universe import BY_CODE


@dataclass
class PositionBook:
    """Signed USD notionals per currency (positive == long the EM currency)."""

    positions: dict[str, float] = field(default_factory=dict)

    def gross_usd(self) -> float:
        return float(sum(abs(v) for v in self.positions.values()))

    def net_usd(self) -> float:
        return float(sum(self.positions.values()))

    def by_region(self) -> dict[str, float]:
        agg: dict[str, float] = {}
        for code, notional in self.positions.items():
            region = BY_CODE[code].region if code in BY_CODE else "Other"
            agg[region] = agg.get(region, 0.0) + notional
        return agg

    def concentration(self) -> float:
        """Largest single-name share of gross exposure (0..1)."""
        gross = self.gross_usd()
        if gross <= 0:
            return 0.0
        return max(abs(v) for v in self.positions.values()) / gross

    def largest_positions(self, n: int = 5) -> list[tuple[str, float]]:
        return sorted(self.positions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]


@dataclass(frozen=True)
class RiskLimits:
    gross_limit_usd: float
    per_ccy_limit_usd: float
    max_concentration: float = 0.40


@dataclass(frozen=True)
class LimitBreach:
    kind: str  # "gross" | "per_ccy" | "concentration"
    detail: str
    value: float
    limit: float

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "value": round(self.value, 4),
            "limit": round(self.limit, 4),
        }


def check_limits(book: PositionBook, limits: RiskLimits) -> list[LimitBreach]:
    """Return all breached limits (empty list == within risk appetite)."""
    breaches: list[LimitBreach] = []

    gross = book.gross_usd()
    if gross > limits.gross_limit_usd:
        breaches.append(LimitBreach("gross", "gross exposure", gross, limits.gross_limit_usd))

    for code, notional in book.positions.items():
        if abs(notional) > limits.per_ccy_limit_usd:
            breaches.append(
                LimitBreach("per_ccy", f"{code} position", abs(notional), limits.per_ccy_limit_usd)
            )

    conc = book.concentration()
    if conc > limits.max_concentration:
        breaches.append(
            LimitBreach("concentration", "single-name concentration", conc, limits.max_concentration)
        )

    return breaches
