"""Crank–Nicolson solver for the Tsiveriotis–Fernandes PDE.

The timestep follows Ayache, Forsyth, and Vetzal, Appendix A.1. The cash
component and the total value are rolled back with theta-weighting (fully
implicit for the first two steps after maturity and after each coupon), then
conversion, put, and call constraints are applied. Each linear system is solved
with ``QuantLib.TridiagonalOperator.solveFor``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import QuantLib as ql

from pricing import (
    ConvertibleParams,
    Greeks,
    _build_schedule,
    _callability,
    _conversion_ratio,
    _dividends,
)


@dataclass(frozen=True)
class _Event:
    time: float
    amount: float
    kind: str  # "coupon", "call", "put"
    trigger: float = 0.0


@dataclass(frozen=True)
class CnGrid:
    """Total value on the market-spot axis after one backward solve."""

    spot: np.ndarray
    npv: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray


def _year_fraction(day_counter: ql.DayCounter, start: ql.Date, end: ql.Date) -> float:
    return float(day_counter.yearFraction(start, end))


def _collect_events(
    params: ConvertibleParams,
    today: ql.Date,
    settlement: ql.Date,
    schedule: ql.Schedule,
    day_counter: ql.DayCounter,
) -> tuple[float, float, list[_Event], list[tuple[float, float]]]:
    """Maturity, redemption, dated events, and discrete dividends, timed from settlement."""
    exercise_date = schedule[-1]
    maturity = _year_fraction(day_counter, settlement, exercise_date)
    conversion_ratio = _conversion_ratio(params)
    callability = _callability(params, schedule)

    bond = ql.ConvertibleFixedCouponBond(
        ql.EuropeanExercise(exercise_date),
        conversion_ratio,
        callability,
        schedule[0],
        params.settlement_days,
        [params.coupon],
        ql.Thirty360(ql.Thirty360.BondBasis),
        schedule,
        params.redemption,
    )
    cashflows = list(bond.cashflows())
    if not cashflows:
        raise ValueError("Convertible bond has no cashflows")
    redemption = float(cashflows[-1].amount())

    events: list[_Event] = []
    for cf in cashflows[:-1]:
        if cf.date() < settlement:
            continue
        events.append(
            _Event(
                _year_fraction(day_counter, settlement, cf.date()),
                float(cf.amount()),
                "coupon",
            )
        )

    # QuantLib-Python does not expose SoftCallability.trigger(), so the
    # schedule specs on ConvertibleParams are the source of the call/put dates.
    strike = params.redemption / conversion_ratio
    if params.enable_soft_calls:
        for call in params.soft_calls:
            when = _year_fraction(day_counter, settlement, schedule[call.year_index])
            events.append(_Event(when, float(call.price), "call", strike * float(call.trigger)))
    if params.enable_puts:
        for put in params.puts:
            when = _year_fraction(day_counter, settlement, schedule[put.year_index])
            events.append(_Event(when, float(put.price), "put"))

    dividends: list[tuple[float, float]] = []
    for div in _dividends(params, today, exercise_date):
        if div.date() < settlement:
            continue
        dividends.append(
            (
                _year_fraction(day_counter, settlement, div.date()),
                float(div.amount()),
            )
        )
    return maturity, redemption, events, dividends


def _escrow(dividends: list[tuple[float, float]], t: float, rate: float) -> float:
    total = 0.0
    for t_div, amount in dividends:
        if t_div + 1e-12 >= t:
            total += amount * float(np.exp(-rate * (t_div - t)))
    return total


def _spatial_coefficients(
    rate: float, dividend: float, vol: float, dx: float
) -> tuple[float, float, float]:
    """Central coefficients of H on a uniform grid in x = log S.

    H = (σ²/2) ∂²/∂x² + (r − q − σ²/2) ∂/∂x.
    """
    variance = vol * vol
    nu = rate - dividend - 0.5 * variance
    half = 0.5 * variance / (dx * dx)
    drift = nu / (2.0 * dx)
    return half - drift, -variance / (dx * dx), half + drift


def _apply_generator(
    values: np.ndarray,
    lower: float,
    diag: float,
    upper: float,
    discount: float,
) -> np.ndarray:
    out = np.empty_like(values)
    out[1:-1] = (
        lower * values[:-2]
        + diag * values[1:-1]
        + upper * values[2:]
        - discount * values[1:-1]
    )
    out[0] = 0.0
    out[-1] = 0.0
    return out


def _solve_tridiagonal(lower: float, diag: float, upper: float, rhs: np.ndarray) -> np.ndarray:
    """Solve a constant-coefficient tridiagonal system with QuantLib.

    Both boundary rows are homogeneous Neumann conditions so a pure cash
    profile (flat in the spot) stays flat. Conversion and call constraints
    are applied after the linear step.
    """
    n = int(rhs.size)
    operator = ql.TridiagonalOperator.identity(n)
    operator.setMidRows(float(lower), float(diag), float(upper))
    operator.setFirstRow(1.0, -1.0)
    operator.setLastRow(-1.0, 1.0)
    solution = operator.solveFor(ql.Array(rhs.tolist()))
    return np.fromiter((solution[i] for i in range(n)), dtype=float, count=n)


def _lhs_coefficients(
    h_lower: float, h_diag: float, h_upper: float, discount: float, dt: float, theta: float
) -> tuple[float, float, float]:
    # (I − θ Δτ L) with L = H − discount.
    scale = theta * dt
    return (
        -scale * h_lower,
        1.0 - scale * h_diag + scale * discount,
        -scale * h_upper,
    )


def _events_at(events: list[_Event], t: float) -> list[_Event]:
    return [event for event in events if abs(event.time - t) <= 1e-8]


def _apply_constraints(
    total: np.ndarray,
    cash: np.ndarray,
    stock: np.ndarray,
    kappa: float,
    events_here: list[_Event],
    convertible: bool,
) -> bool:
    """Project call, put, coupon, and conversion. Returns whether a coupon was paid."""
    coupon_added = False
    parity = kappa * stock
    for event in events_here:
        if event.kind == "call" and event.trigger > 0.0:
            active = stock >= event.trigger
            cap = np.maximum(event.amount, parity)
            binds = active & (total > cap)
            take_shares = parity >= event.amount
            np.copyto(total, np.where(binds, cap, total))
            np.copyto(cash, np.where(binds & take_shares, 0.0, cash))
            np.copyto(cash, np.where(binds & ~take_shares, event.amount, cash))
        elif event.kind == "put":
            binds = total < event.amount
            np.copyto(total, np.where(binds, event.amount, total))
            np.copyto(cash, np.where(binds, event.amount, cash))
    for event in events_here:
        if event.kind == "coupon":
            total += event.amount
            cash += event.amount
            coupon_added = True
    if convertible:
        binds = total <= parity
        np.copyto(total, np.where(binds, parity, total))
        np.copyto(cash, np.where(binds, 0.0, cash))
    np.minimum(cash, total, out=cash)
    return coupon_added


def _theta_step(
    total: np.ndarray,
    cash: np.ndarray,
    h_lower: float,
    h_diag: float,
    h_upper: float,
    rate: float,
    spread: float,
    dt: float,
    theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One AFV theta step of the cash component and the total value."""
    rhs_cash = cash + (1.0 - theta) * dt * _apply_generator(
        cash, h_lower, h_diag, h_upper, rate + spread
    )
    rhs_cash[0] = 0.0
    rhs_cash[-1] = 0.0
    cash_l, cash_d, cash_u = _lhs_coefficients(
        h_lower, h_diag, h_upper, rate + spread, dt, theta
    )
    new_cash = _solve_tridiagonal(cash_l, cash_d, cash_u, rhs_cash)

    rhs_total = total + (1.0 - theta) * dt * _apply_generator(
        total, h_lower, h_diag, h_upper, rate
    )
    # Calendar-time PDE is V_t + HV − r V − p B = 0, so the τ-source is −p B.
    # With V = B this discounts the cash profile at r + p.
    rhs_total[1:-1] -= dt * spread * (theta * new_cash[1:-1] + (1.0 - theta) * cash[1:-1])
    rhs_total[0] = 0.0
    rhs_total[-1] = 0.0
    total_l, total_d, total_u = _lhs_coefficients(h_lower, h_diag, h_upper, rate, dt, theta)
    new_total = _solve_tridiagonal(total_l, total_d, total_u, rhs_total)
    return new_total, new_cash


def solve_cn(
    params: ConvertibleParams,
    spot: float | None = None,
    volatility: float | None = None,
    credit_spread: float | None = None,
    n_space: int | None = None,
    spot_min: float | None = None,
    spot_max: float | None = None,
) -> CnGrid:
    """Roll the TF system back to settlement and return value, delta, and gamma vs spot."""
    spot_val = float(params.spot if spot is None else spot)
    vol = float(params.volatility if volatility is None else volatility)
    spread = float(params.credit_spread if credit_spread is None else credit_spread)
    if spot_val <= 0.0:
        raise ValueError("Spot must be positive")
    if vol < 0.0:
        raise ValueError("Volatility must be non-negative")
    if params.time_steps < 2:
        raise ValueError("time_steps must be at least 2")

    schedule, today, settlement, _issue = _build_schedule(params)
    day_counter = ql.Actual365Fixed()
    maturity, redemption, events, dividends = _collect_events(
        params, today, settlement, schedule, day_counter
    )
    if maturity <= 0.0:
        raise ValueError("Convertible maturity must be after settlement")

    rate = float(params.risk_free_rate)
    dividend_yield = float(params.dividend_yield)
    kappa = _conversion_ratio(params)
    escrow0 = _escrow(dividends, 0.0, rate)
    stripped = spot_val - escrow0
    if stripped <= 0.0:
        raise ValueError("Spot does not cover the present value of discrete dividends")

    width = max(1.0, vol * np.sqrt(maturity) * 6.0)
    covered = [stripped]
    for market in (spot_min, spot_max):
        if market is not None and market - escrow0 > 0.0:
            covered.append(market - escrow0)
    parity_x = params.redemption / kappa - escrow0
    if parity_x > 0.0:
        covered.append(parity_x)
    x_min = min(covered) * float(np.exp(-width))
    x_max = max(covered) * float(np.exp(width))
    x_min = max(x_min, 1e-4)

    n = 301 if n_space is None else n_space
    n = max(151, n)
    stock = np.exp(np.linspace(np.log(x_min), np.log(x_max), n))
    dx = float(np.log(stock[1] / stock[0]))
    h_lower, h_diag, h_upper = _spatial_coefficients(rate, dividend_yield, vol, dx)

    base = np.linspace(0.0, maturity, int(params.time_steps) + 1)
    event_times = np.array(
        [event.time for event in events if 0.0 < event.time < maturity], dtype=float
    )
    times = np.unique(np.concatenate([base, event_times, [0.0, maturity]]))

    total = np.full(n, redemption, dtype=float)
    cash = np.full(n, redemption, dtype=float)
    _apply_constraints(
        total,
        cash,
        stock + _escrow(dividends, maturity, rate),
        kappa,
        _events_at(events, maturity),
        True,
    )

    implicit_left = 2
    for index in range(len(times) - 2, -1, -1):
        t_new = float(times[index])
        dt = float(times[index + 1] - t_new)
        if dt <= 0.0:
            continue
        theta = 1.0 if implicit_left > 0 else 0.5
        if implicit_left > 0:
            implicit_left -= 1
        escrow_new = _escrow(dividends, t_new, rate)
        total, cash = _theta_step(
            total,
            cash,
            h_lower,
            h_diag,
            h_upper,
            rate,
            spread,
            dt,
            theta,
        )
        convertible = params.american or abs(t_new - maturity) <= 1e-8
        coupon_added = _apply_constraints(
            total,
            cash,
            stock + escrow_new,
            kappa,
            _events_at(events, t_new),
            convertible,
        )
        if coupon_added:
            implicit_left = 2

    market_spot = stock + escrow0
    delta = np.gradient(total, market_spot)
    gamma = np.gradient(delta, market_spot)
    return CnGrid(spot=market_spot, npv=total, delta=delta, gamma=gamma)


def _sample(grid: CnGrid, spot: float) -> tuple[float, float, float]:
    if spot < grid.spot[0] or spot > grid.spot[-1]:
        raise ValueError(
            f"Spot {spot} is outside the finite-difference grid "
            f"[{grid.spot[0]:.4f}, {grid.spot[-1]:.4f}]"
        )
    npv = float(np.interp(spot, grid.spot, grid.npv))
    delta = float(np.interp(spot, grid.spot, grid.delta))
    gamma = float(np.interp(spot, grid.spot, grid.gamma))
    return npv, delta, gamma


def price_npv_cn(
    params: ConvertibleParams,
    spot: float | None = None,
    volatility: float | None = None,
    credit_spread: float | None = None,
    n_space: int | None = None,
) -> float:
    grid = solve_cn(
        params,
        spot=spot,
        volatility=volatility,
        credit_spread=credit_spread,
        n_space=n_space,
    )
    return _sample(grid, params.spot if spot is None else spot)[0]


def greeks_cn(
    params: ConvertibleParams,
    spot_min: float | None = None,
    spot_max: float | None = None,
    n_space: int | None = None,
) -> tuple[Greeks, CnGrid]:
    """Headline Greeks. Delta and gamma are grid derivatives; vega and CS01 are bumps."""
    grid = solve_cn(params, n_space=n_space, spot_min=spot_min, spot_max=spot_max)
    npv, delta, gamma = _sample(grid, params.spot)
    bumped_vol = _sample(
        solve_cn(
            params,
            volatility=params.volatility + params.vol_bump,
            n_space=n_space,
            spot_min=spot_min,
            spot_max=spot_max,
        ),
        params.spot,
    )[0]
    bumped_spread = _sample(
        solve_cn(
            params,
            credit_spread=params.credit_spread + params.spread_bump,
            n_space=n_space,
            spot_min=spot_min,
            spot_max=spot_max,
        ),
        params.spot,
    )[0]
    vega = (bumped_vol - npv) / params.vol_bump
    credit_spread_01 = (bumped_spread - npv) / params.spread_bump * 1.0e-4
    return (
        Greeks(
            npv=npv,
            delta=delta,
            gamma=gamma,
            vega=vega,
            credit_spread_01=credit_spread_01,
        ),
        grid,
    )


def spot_frame(grid: CnGrid, spot_min: float, spot_max: float, n_points: int) -> pd.DataFrame:
    """Interpolate a solved grid onto a uniform market-spot window."""
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    if spot_min <= 0.0 or spot_max <= spot_min:
        raise ValueError("Require 0 < spot_min < spot_max")
    spots = np.linspace(spot_min, spot_max, n_points)
    rows = []
    for s in spots:
        npv, delta, gamma = _sample(grid, float(s))
        rows.append({"spot": float(s), "npv": npv, "delta": delta, "gamma": gamma})
    return pd.DataFrame(rows)
