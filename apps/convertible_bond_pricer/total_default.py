"""Total-default convertible from Ayache, Forsyth, and Vetzal, Section 6.

With a jump of the stock to zero and no recovery, the hedge model is the
ordinary convertible PDE with the short rate replaced by r + p. Table 1 of
the paper is hard-wired: face 100, conversion ratio 1, coupons of 4 every
half year, a continuous call at clean 110 after year 2, and a put at clean
105 only at year 3.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import QuantLib as ql


FACE = 100.0
COUPON = 4.0
COUPON_TIMES = tuple(0.5 * k for k in range(1, 11))  # 0.5, 1.0, ..., 5.0
MATURITY = 5.0
RATE = 0.05
HAZARD = 0.02
VOLATILITY = 0.20
CALL_CLEAN = 110.0
CALL_START = 2.0
PUT_CLEAN = 105.0
PUT_TIME = 3.0
TABLE_SPOT = 100.0

# Table 2, total-default column, V(100) at t = 0.
TABLE2_TOTAL_DEFAULT = {
    200: 122.7341,
    400: 122.7333,
    800: 122.7325,
    1600: 122.7319,
    3200: 122.7316,
}


@dataclass(frozen=True)
class TotalDefaultGrid:
    spot: np.ndarray
    npv: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray


def _spot_operator(n: int, s_max: float, rate: float):
    """Central coefficients of the spot-space operator on a uniform S grid.

    The grid runs from 0 to ``s_max``. Rows 0 and n-1 are overwritten by the
    boundary conditions, so their coefficients are unused.
    """
    stock = np.linspace(0.0, s_max, n)
    spacing = float(stock[1] - stock[0])
    diffusion = 0.5 * VOLATILITY * VOLATILITY * stock * stock
    convection = rate * stock
    return (
        stock,
        diffusion / spacing**2 - convection / (2.0 * spacing),
        -2.0 * diffusion / spacing**2,
        diffusion / spacing**2 + convection / (2.0 * spacing),
    )


def _apply_operator(values, lower, diag, upper, discount):
    out = np.empty_like(values)
    out[1:-1] = (
        lower[1:-1] * values[:-2]
        + (diag[1:-1] - discount) * values[1:-1]
        + upper[1:-1] * values[2:]
    )
    out[0] = 0.0
    out[-1] = 0.0
    return out


def _lhs(lower, diag, upper, discount, dt, theta):
    scale = theta * dt
    return -scale * lower, 1.0 - scale * diag + scale * discount, -scale * upper


def _solve_tridiagonal(lower, diag, upper, rhs, penalty_diag=None, penalty_rhs=None):
    """Flat Neumann at S = 0, Dirichlet at the high spot (rhs[-1])."""
    n = int(rhs.size)
    operator = ql.TridiagonalOperator.identity(n)
    operator.setFirstRow(1.0, -1.0)
    operator.setLastRow(0.0, 1.0)
    solved = rhs if penalty_diag is None else rhs.copy()
    for i in range(1, n - 1):
        extra = 0.0 if penalty_diag is None else float(penalty_diag[i])
        operator.setMidRow(i, float(lower[i]), float(diag[i]) + extra, float(upper[i]))
        if extra != 0.0:
            solved[i] += float(penalty_rhs[i])
    solution = operator.solveFor(ql.Array(solved.tolist()))
    return np.fromiter((solution[i] for i in range(n)), dtype=float, count=n)


def _is_coupon(t: float) -> bool:
    return any(abs(t - coupon_time) <= 1e-8 for coupon_time in COUPON_TIMES)


def accrued(t: float, on_coupon: bool) -> float:
    """Accrued interest on the pending coupon.

    On a coupon date the constraint is applied to the cum-coupon value, so
    accrued interest is the coupon itself.
    """
    if on_coupon or t >= MATURITY - 1e-12:
        return COUPON
    if t <= 0.0:
        return 0.0
    later = [coupon_time for coupon_time in COUPON_TIMES if coupon_time > t + 1e-12]
    t_next = later[0]
    earlier = [coupon_time for coupon_time in COUPON_TIMES if coupon_time < t_next - 1e-12]
    t_prev = earlier[-1] if earlier and earlier[-1] < t else 0.0
    return COUPON * (t - t_prev) / (t_next - t_prev)


def _high_spot(t: float, spot: float, rate: float, include_coupon_now: bool) -> float:
    """Deep in-the-money value.

    Until the call starts, the holder keeps each protected coupon and is left
    with the stock once the call window opens. Inside the window the issuer
    forces the value down to the stock.
    """
    if t > CALL_START + 1e-8:
        return spot
    premium = 0.0
    for coupon_time in COUPON_TIMES:
        if coupon_time > CALL_START + 1e-12 or coupon_time < t - 1e-12:
            continue
        if abs(coupon_time - t) <= 1e-12 and not include_coupon_now:
            continue
        premium += COUPON * math.exp(-rate * max(coupon_time - t, 0.0))
    return spot + premium


def _obstacle(stock, t, on_coupon):
    """Lower and upper bounds on the cum-coupon value."""
    interest = accrued(t, on_coupon)
    lower = stock.copy()
    if abs(t - PUT_TIME) <= 1e-8:
        np.maximum(lower, PUT_CLEAN + interest, out=lower)
    if t > CALL_START + 1e-8:
        upper = np.maximum(CALL_CLEAN + interest, stock)
    else:
        upper = np.full_like(stock, np.inf)
    return lower, upper


def solve_total_default(n: int, s_max: float = 2000.0, rate: float | None = None) -> TotalDefaultGrid:
    """Roll the Table 1 total-default contract back to t = 0.

    ``n`` is the number of spot nodes and the number of time steps. ``rate``
    replaces r + p in both the drift and the discount; omit it for the
    total-default column.
    """
    if n < 20:
        raise ValueError("n must be at least 20")
    if s_max <= 0.0:
        raise ValueError("s_max must be positive")

    discount = RATE + HAZARD if rate is None else rate
    stock, h_lower, h_diag, h_upper = _spot_operator(n, s_max, discount)

    base = np.linspace(0.0, MATURITY, n + 1)
    times = np.unique(np.round(np.concatenate([base, COUPON_TIMES, [CALL_START, PUT_TIME]]), 10))

    values = np.maximum(stock, FACE + COUPON)
    values[-1] = stock[-1]
    implicit_left = 2

    for index in range(len(times) - 2, -1, -1):
        t_new = float(times[index])
        dt = float(times[index + 1] - t_new)
        if dt <= 0.0:
            continue
        theta = 1.0 if implicit_left > 0 else 0.5
        if implicit_left > 0:
            implicit_left -= 1

        on_coupon = _is_coupon(t_new) and t_new < MATURITY - 1e-8
        cash = COUPON if on_coupon else 0.0
        lower_bound, upper_bound = _obstacle(stock, t_new, on_coupon)
        lower_ex = lower_bound - cash
        upper_ex = upper_bound - cash

        rhs = values + (1.0 - theta) * dt * _apply_operator(
            values, h_lower, h_diag, h_upper, discount
        )
        rhs[0] = 0.0
        rhs[-1] = _high_spot(t_new, float(stock[-1]), discount, False)
        band_lower, band_diag, band_upper = _lhs(h_lower, h_diag, h_upper, discount, dt, theta)
        values = _solve_tridiagonal(band_lower, band_diag, band_upper, rhs)

        # Pin nodes that breach conversion, the put, or the call, and re-solve
        # so the obstacle is felt by the neighbouring continuation nodes.
        penalty = 1.0e7
        for _ in range(8):
            too_low = values[1:-1] < lower_ex[1:-1] - 1.0e-7
            too_high = values[1:-1] > upper_ex[1:-1] + 1.0e-7
            if not np.any(too_low) and not np.any(too_high):
                break
            penalty_diag = np.zeros_like(values)
            penalty_rhs = np.zeros_like(values)
            interior = np.arange(1, n - 1)
            low_nodes = interior[too_low]
            high_nodes = interior[too_high]
            penalty_diag[low_nodes] = penalty
            penalty_rhs[low_nodes] = penalty * lower_ex[low_nodes]
            penalty_diag[high_nodes] = penalty
            penalty_rhs[high_nodes] = penalty * upper_ex[high_nodes]
            values = _solve_tridiagonal(
                band_lower, band_diag, band_upper, rhs, penalty_diag, penalty_rhs
            )

        if on_coupon:
            values += COUPON
            implicit_left = 2
        values[-1] = _high_spot(t_new, float(stock[-1]), discount, on_coupon)

    delta = np.gradient(values, stock)
    gamma = np.gradient(delta, stock)
    return TotalDefaultGrid(spot=stock, npv=values, delta=delta, gamma=gamma)


def value_at(grid: TotalDefaultGrid, spot: float) -> tuple[float, float, float]:
    if spot < grid.spot[0] or spot > grid.spot[-1]:
        raise ValueError(
            f"Spot {spot} is outside the grid [{grid.spot[0]:.4g}, {grid.spot[-1]:.4g}]"
        )
    return (
        float(np.interp(spot, grid.spot, grid.npv)),
        float(np.interp(spot, grid.spot, grid.delta)),
        float(np.interp(spot, grid.spot, grid.gamma)),
    )


def price_at_table_spot(n: int) -> float:
    return value_at(solve_total_default(n), TABLE_SPOT)[0]


def sample_frame(grid: TotalDefaultGrid, spot_min: float, spot_max: float, n_points: int) -> pd.DataFrame:
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    if spot_min < 0.0 or spot_max <= spot_min:
        raise ValueError("Require 0 <= spot_min < spot_max")
    spots = np.linspace(spot_min, spot_max, n_points)
    rows = []
    for spot in spots:
        npv, delta, gamma = value_at(grid, float(spot))
        rows.append({"spot": float(spot), "npv": npv, "delta": delta, "gamma": gamma})
    return pd.DataFrame(rows)
