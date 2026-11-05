"""Convertible fixed-coupon bond pricing helpers (QuantLib binomial TF)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import QuantLib as ql


TREE_TYPES = {
    "Cox-Ross-Rubinstein": "crr",
    "Jarrow-Rudd": "jr",
    "Additive EQP": "eqp",
    "Trigeorgis": "trigeorgis",
    "Tian": "tian",
    "Leisen-Reimer": "lr",
    "Joshi4": "j4",
}

FREQUENCY_MAP = {
    "Annual": ql.Annual,
    "Semiannual": ql.Semiannual,
    "Quarterly": ql.Quarterly,
}


@dataclass(frozen=True)
class SoftCallSpec:
    """Soft call at schedule index ``year_index`` (annual schedule ≈ years from issue)."""

    year_index: int
    price: float
    trigger: float = 1.20


@dataclass(frozen=True)
class PutSpec:
    year_index: int
    price: float


@dataclass(frozen=True)
class ConvertibleParams:
    evaluation_date: ql.Date
    spot: float = 36.0
    volatility: float = 0.20
    dividend_yield: float = 0.02
    risk_free_rate: float = 0.06
    credit_spread: float = 0.005
    coupon: float = 0.05
    frequency: str = "Annual"
    tenor_years: int = 5
    conversion_ratio: float | None = None  # None => ATM = redemption / spot
    redemption: float = 100.0
    settlement_days: int = 3
    american: bool = True
    tree_type: str = "Cox-Ross-Rubinstein"
    time_steps: int = 201
    soft_calls: tuple[SoftCallSpec, ...] = (
        SoftCallSpec(2, 101.5, 1.20),
        SoftCallSpec(4, 100.85, 1.20),
    )
    puts: tuple[PutSpec, ...] = (PutSpec(3, 105.0),)
    enable_soft_calls: bool = True
    enable_puts: bool = True
    dividend_amount: float = 1.0
    dividend_period_months: int = 6
    enable_discrete_dividends: bool = True
    bump_rel: float = 0.01
    vol_bump: float = 0.01
    spread_bump: float = 0.0001  # 1 bp


@dataclass(frozen=True)
class Greeks:
    npv: float
    delta: float
    gamma: float
    vega: float
    credit_spread_01: float


def _as_date(d) -> ql.Date:
    if isinstance(d, ql.Date):
        return d
    return ql.Date(d.day, d.month, d.year)


def _calendar() -> ql.Calendar:
    return ql.TARGET()


def _build_schedule(params: ConvertibleParams) -> tuple[ql.Schedule, ql.Date, ql.Date, ql.Date]:
    calendar = _calendar()
    today = calendar.adjust(_as_date(params.evaluation_date))
    ql.Settings.instance().evaluationDate = today
    settlement_date = calendar.advance(today, params.settlement_days, ql.Days)
    exercise_date = calendar.advance(settlement_date, params.tenor_years, ql.Years)
    issue_date = calendar.advance(exercise_date, -params.tenor_years, ql.Years)
    frequency = FREQUENCY_MAP[params.frequency]
    schedule = ql.Schedule(
        issue_date,
        exercise_date,
        ql.Period(frequency),
        calendar,
        ql.ModifiedFollowing,
        ql.ModifiedFollowing,
        ql.DateGeneration.Backward,
        False,
    )
    return schedule, today, settlement_date, issue_date


def _callability(params: ConvertibleParams, schedule: ql.Schedule) -> ql.CallabilitySchedule:
    schedule_len = len(schedule)
    callability = ql.CallabilitySchedule()
    if params.enable_soft_calls:
        for call in params.soft_calls:
            if call.year_index < 0 or call.year_index >= schedule_len:
                raise ValueError(
                    f"Soft-call schedule index {call.year_index} out of range "
                    f"[0, {schedule_len - 1}]"
                )
            callability.append(
                ql.SoftCallability(
                    ql.BondPrice(call.price, ql.BondPrice.Clean),
                    schedule[call.year_index],
                    call.trigger,
                )
            )
    if params.enable_puts:
        for put in params.puts:
            if put.year_index < 0 or put.year_index >= schedule_len:
                raise ValueError(
                    f"Put schedule index {put.year_index} out of range "
                    f"[0, {schedule_len - 1}]"
                )
            callability.append(
                ql.Callability(
                    ql.BondPrice(put.price, ql.BondPrice.Clean),
                    ql.Callability.Put,
                    schedule[put.year_index],
                )
            )
    return callability


def _dividends(
    params: ConvertibleParams, today: ql.Date, exercise_date: ql.Date
) -> ql.DividendSchedule:
    dividends = ql.DividendSchedule()
    if not params.enable_discrete_dividends or params.dividend_amount == 0.0:
        return dividends
    period = ql.Period(params.dividend_period_months, ql.Months)
    d = today + period
    while d < exercise_date:
        dividends.append(ql.FixedDividend(params.dividend_amount, d))
        d += period
    return dividends


def _conversion_ratio(params: ConvertibleParams) -> float:
    if params.conversion_ratio is None:
        if params.spot <= 0.0:
            raise ValueError("Spot must be positive for ATM conversion ratio")
        return params.redemption / params.spot
    return params.conversion_ratio


def _make_process(
    settlement_date: ql.Date,
    spot_quote: ql.SimpleQuote,
    dividend_yield: float,
    risk_free_rate: float,
    volatility: float,
) -> ql.BlackScholesMertonProcess:
    calendar = _calendar()
    day_counter = ql.Actual365Fixed()
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(spot_quote),
        ql.YieldTermStructureHandle(
            ql.FlatForward(settlement_date, dividend_yield, day_counter)
        ),
        ql.YieldTermStructureHandle(
            ql.FlatForward(settlement_date, risk_free_rate, day_counter)
        ),
        ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(settlement_date, calendar, volatility, day_counter)
        ),
    )


def _tree_key(tree_type: str) -> str:
    if tree_type in TREE_TYPES:
        return TREE_TYPES[tree_type]
    key = tree_type.lower()
    if key in TREE_TYPES.values():
        return key
    raise ValueError(f"Unknown tree type: {tree_type}")


def price_npv(
    params: ConvertibleParams,
    spot: float | None = None,
    volatility: float | None = None,
    credit_spread: float | None = None,
) -> float:
    """Price the convertible; optional overrides for bump-and-reprice."""
    spot_val = params.spot if spot is None else spot
    vol_val = params.volatility if volatility is None else volatility
    spread_val = params.credit_spread if credit_spread is None else credit_spread

    if spot_val <= 0.0:
        raise ValueError("Spot must be positive")
    if params.time_steps <= 0:
        raise ValueError("time_steps must be positive")
    if params.frequency not in FREQUENCY_MAP:
        raise ValueError(f"Unsupported frequency: {params.frequency}")

    schedule, today, settlement_date, issue_date = _build_schedule(params)
    exercise_date = schedule[-1]
    callability = _callability(params, schedule)
    dividends = _dividends(params, today, exercise_date)
    conversion_ratio = _conversion_ratio(params)

    if params.american:
        exercise = ql.AmericanExercise(settlement_date, exercise_date)
    else:
        exercise = ql.EuropeanExercise(exercise_date)

    bond = ql.ConvertibleFixedCouponBond(
        exercise,
        conversion_ratio,
        callability,
        issue_date,
        params.settlement_days,
        [params.coupon],
        ql.Thirty360(ql.Thirty360.BondBasis),
        schedule,
        params.redemption,
    )

    spot_quote = ql.SimpleQuote(spot_val)
    process = _make_process(
        settlement_date,
        spot_quote,
        params.dividend_yield,
        params.risk_free_rate,
        vol_val,
    )
    credit = ql.QuoteHandle(ql.SimpleQuote(spread_val))
    engine = ql.BinomialConvertibleEngine(
        process, _tree_key(params.tree_type), params.time_steps, credit, dividends
    )
    bond.setPricingEngine(engine)
    return float(bond.NPV())


def compute_greeks(params: ConvertibleParams) -> Greeks:
    """Central FD delta/gamma on spot; forward FD vega and credit-spread 01."""
    s = params.spot
    h = params.bump_rel * s
    if h <= 0.0:
        raise ValueError("Relative bump must be positive")

    v0 = price_npv(params, spot=s)
    v_up = price_npv(params, spot=s + h)
    v_dn = price_npv(params, spot=s - h)
    delta = (v_up - v_dn) / (2.0 * h)
    gamma = (v_up - 2.0 * v0 + v_dn) / (h * h)

    v_vol = price_npv(params, volatility=params.volatility + params.vol_bump)
    vega = (v_vol - v0) / params.vol_bump

    v_spread = price_npv(
        params, credit_spread=params.credit_spread + params.spread_bump
    )
    credit_spread_01 = (v_spread - v0) / params.spread_bump * 1.0e-4

    return Greeks(
        npv=v0,
        delta=delta,
        gamma=gamma,
        vega=vega,
        credit_spread_01=credit_spread_01,
    )


def spot_grid(
    params: ConvertibleParams,
    spot_min: float,
    spot_max: float,
    n_points: int,
) -> pd.DataFrame:
    """NPV, delta, and gamma across a spot grid (FD at each point)."""
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    if spot_min <= 0.0 or spot_max <= spot_min:
        raise ValueError("Require 0 < spot_min < spot_max")

    spots = np.linspace(spot_min, spot_max, n_points)
    rows = []
    h_rel = params.bump_rel
    for s in spots:
        h = h_rel * s
        v0 = price_npv(params, spot=float(s))
        v_up = price_npv(params, spot=float(s + h))
        v_dn = price_npv(params, spot=float(s - h))
        rows.append(
            {
                "spot": float(s),
                "npv": v0,
                "delta": (v_up - v_dn) / (2.0 * h),
                "gamma": (v_up - 2.0 * v0 + v_dn) / (h * h),
            }
        )
    return pd.DataFrame(rows)


def default_params(evaluation_date: ql.Date | None = None) -> ConvertibleParams:
    calendar = _calendar()
    if evaluation_date is None:
        evaluation_date = calendar.adjust(ql.Date.todaysDate())
    else:
        evaluation_date = calendar.adjust(_as_date(evaluation_date))
    return ConvertibleParams(evaluation_date=evaluation_date)


def parse_soft_calls(
    year_indices: Sequence[int],
    prices: Sequence[float],
    trigger: float,
) -> tuple[SoftCallSpec, ...]:
    if len(year_indices) != len(prices):
        raise ValueError("Soft-call years and prices must have the same length")
    return tuple(
        SoftCallSpec(int(y), float(p), float(trigger))
        for y, p in zip(year_indices, prices)
    )


def parse_puts(
    year_indices: Sequence[int], prices: Sequence[float]
) -> tuple[PutSpec, ...]:
    if len(year_indices) != len(prices):
        raise ValueError("Put years and prices must have the same length")
    return tuple(PutSpec(int(y), float(p)) for y, p in zip(year_indices, prices))


def parse_int_list(text: str) -> list[int]:
    text = text.strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> list[float]:
    text = text.strip()
    if not text:
        return []
    return [float(x.strip()) for x in text.split(",") if x.strip()]
