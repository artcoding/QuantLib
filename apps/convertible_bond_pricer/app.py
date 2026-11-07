"""Streamlit UI for convertible fixed-coupon bond pricing."""

from __future__ import annotations

import datetime as dt

import plotly.graph_objects as go
import QuantLib as ql
import streamlit as st

from pricing import (
    TREE_TYPES,
    ConvertibleParams,
    PutSpec,
    SoftCallSpec,
    compute_greeks,
    parse_float_list,
    parse_int_list,
    parse_puts,
    parse_soft_calls,
    spot_grid,
)
from tf_cn import greeks_cn, spot_frame
from total_default import TABLE_SPOT, sample_frame, solve_total_default, value_at

SCHEMES = ("Binomial TF", "Crank-Nicolson TF", "Ayache total default")


def _params_to_tuple(params: ConvertibleParams) -> tuple:
    ed = params.evaluation_date
    return (
        (ed.year(), ed.month(), ed.dayOfMonth()),
        params.spot,
        params.volatility,
        params.dividend_yield,
        params.risk_free_rate,
        params.credit_spread,
        params.coupon,
        params.frequency,
        params.tenor_years,
        params.conversion_ratio,
        params.redemption,
        params.settlement_days,
        params.american,
        params.tree_type,
        params.time_steps,
        tuple((c.year_index, c.price, c.trigger) for c in params.soft_calls),
        tuple((p.year_index, p.price) for p in params.puts),
        params.enable_soft_calls,
        params.enable_puts,
        params.dividend_amount,
        params.dividend_period_months,
        params.enable_discrete_dividends,
        params.bump_rel,
        params.vol_bump,
        params.spread_bump,
    )


def _params_from_tuple(key: tuple) -> ConvertibleParams:
    (
        ed,
        spot,
        volatility,
        dividend_yield,
        risk_free_rate,
        credit_spread,
        coupon,
        frequency,
        tenor_years,
        conversion_ratio,
        redemption,
        settlement_days,
        american,
        tree_type,
        time_steps,
        soft_calls,
        puts,
        enable_soft_calls,
        enable_puts,
        dividend_amount,
        dividend_period_months,
        enable_discrete_dividends,
        bump_rel,
        vol_bump,
        spread_bump,
    ) = key
    return ConvertibleParams(
        evaluation_date=ql.Date(ed[2], ed[1], ed[0]),
        spot=spot,
        volatility=volatility,
        dividend_yield=dividend_yield,
        risk_free_rate=risk_free_rate,
        credit_spread=credit_spread,
        coupon=coupon,
        frequency=frequency,
        tenor_years=tenor_years,
        conversion_ratio=conversion_ratio,
        redemption=redemption,
        settlement_days=settlement_days,
        american=american,
        tree_type=tree_type,
        time_steps=time_steps,
        soft_calls=tuple(SoftCallSpec(*c) for c in soft_calls),
        puts=tuple(PutSpec(*p) for p in puts),
        enable_soft_calls=enable_soft_calls,
        enable_puts=enable_puts,
        dividend_amount=dividend_amount,
        dividend_period_months=dividend_period_months,
        enable_discrete_dividends=enable_discrete_dividends,
        bump_rel=bump_rel,
        vol_bump=vol_bump,
        spread_bump=spread_bump,
    )


def _greeks_dict(greeks) -> dict:
    return {
        "npv": greeks.npv,
        "delta": greeks.delta,
        "gamma": greeks.gamma,
        "vega": greeks.vega,
        "credit_spread_01": greeks.credit_spread_01,
    }


@st.cache_data(show_spinner=False)
def _cached_binomial(key: tuple, spot_min: float, spot_max: float, n_points: int):
    params = _params_from_tuple(key)
    return _greeks_dict(compute_greeks(params)), spot_grid(params, spot_min, spot_max, n_points)


@st.cache_data(show_spinner=False)
def _cached_total_default(n: int, spot_min: float, spot_max: float, n_points: int):
    grid = solve_total_default(n)
    npv, delta, gamma = value_at(grid, TABLE_SPOT)
    frame = sample_frame(grid, spot_min, spot_max, n_points)
    return {"npv": npv, "delta": delta, "gamma": gamma}, frame


@st.cache_data(show_spinner=False)
def _cached_crank_nicolson(key: tuple, spot_min: float, spot_max: float, n_points: int):
    params = _params_from_tuple(key)
    greeks, grid = greeks_cn(params, spot_min=spot_min, spot_max=spot_max)
    return _greeks_dict(greeks), spot_frame(grid, spot_min, spot_max, n_points)


st.set_page_config(
    page_title="Convertible Bond Pricer",
    page_icon="📈",
    layout="wide",
)


def _ql_date(d: dt.date) -> ql.Date:
    return ql.Date(d.day, d.month, d.year)


def _build_params_from_sidebar() -> ConvertibleParams:
    st.sidebar.header("Market")
    spot = st.sidebar.number_input("Spot", min_value=0.01, value=36.0, step=0.5)
    volatility = st.sidebar.number_input(
        "Volatility", min_value=0.0, value=0.20, step=0.01, format="%.4f"
    )
    dividend_yield = st.sidebar.number_input(
        "Dividend yield", min_value=0.0, value=0.02, step=0.001, format="%.4f"
    )
    risk_free_rate = st.sidebar.number_input(
        "Risk-free rate", min_value=-0.05, value=0.06, step=0.001, format="%.4f"
    )
    credit_spread = st.sidebar.number_input(
        "Credit spread", min_value=0.0, value=0.005, step=0.0005, format="%.5f"
    )

    st.sidebar.header("Bond")
    eval_default = dt.date.today()
    evaluation_date = st.sidebar.date_input("Evaluation date", value=eval_default)
    coupon = st.sidebar.number_input(
        "Coupon rate", min_value=0.0, value=0.05, step=0.001, format="%.4f"
    )
    frequency = st.sidebar.selectbox("Frequency", list(["Annual", "Semiannual", "Quarterly"]))
    tenor_years = st.sidebar.number_input("Tenor (years)", min_value=1, value=5, step=1)
    redemption = st.sidebar.number_input("Redemption", min_value=0.01, value=100.0, step=1.0)
    settlement_days = st.sidebar.number_input("Settlement days", min_value=0, value=3, step=1)
    atm_conversion = st.sidebar.checkbox("ATM conversion ratio (redemption / spot)", value=True)
    conversion_ratio = None
    if not atm_conversion:
        conversion_ratio = st.sidebar.number_input(
            "Conversion ratio",
            min_value=0.0001,
            value=float(redemption / spot),
            step=0.01,
            format="%.6f",
        )

    st.sidebar.header("Soft call / put")
    enable_soft_calls = st.sidebar.checkbox("Enable soft calls", value=True)
    soft_call_years_text = st.sidebar.text_input("Soft-call schedule indices", value="2, 4")
    soft_call_prices_text = st.sidebar.text_input("Soft-call clean prices", value="101.5, 100.85")
    soft_call_trigger = st.sidebar.number_input(
        "Soft-call trigger", min_value=0.0, value=1.20, step=0.05, format="%.2f"
    )
    enable_puts = st.sidebar.checkbox("Enable puts", value=True)
    put_years_text = st.sidebar.text_input("Put schedule indices", value="3")
    put_prices_text = st.sidebar.text_input("Put clean prices", value="105.0")

    st.sidebar.header("Discrete dividends")
    enable_discrete_dividends = st.sidebar.checkbox("Enable discrete dividends", value=True)
    dividend_amount = st.sidebar.number_input(
        "Dividend amount", min_value=0.0, value=1.0, step=0.1
    )
    dividend_period_months = st.sidebar.number_input(
        "Dividend period (months)", min_value=1, value=6, step=1
    )

    st.sidebar.header("Engine")
    scheme = st.sidebar.selectbox("Scheme", list(SCHEMES))
    if scheme == "Ayache total default":
        ayache_nodes = st.sidebar.select_slider(
            "Spot nodes", options=[200, 400, 800], value=200
        )
    else:
        ayache_nodes = 200
    american = st.sidebar.selectbox("Exercise", ["American", "European"]) == "American"
    if scheme == "Binomial TF":
        tree_type = st.sidebar.selectbox("Tree type", list(TREE_TYPES.keys()))
    else:
        tree_type = "Cox-Ross-Rubinstein"
    time_steps = st.sidebar.slider("Time steps", min_value=51, max_value=801, value=201, step=50)
    bump_rel = st.sidebar.number_input(
        "Spot bump (relative)",
        min_value=0.0001,
        value=0.01,
        step=0.001,
        format="%.4f",
        help="Binomial delta and gamma use this bump. Crank–Nicolson delta and gamma are derivatives of the finite-difference grid.",
    )

    st.sidebar.header("Chart")
    spot_pct_min = st.sidebar.number_input("Spot min (% of spot)", value=50.0, step=5.0)
    spot_pct_max = st.sidebar.number_input("Spot max (% of spot)", value=150.0, step=5.0)
    n_points = st.sidebar.slider("Grid points", min_value=5, max_value=61, value=21, step=2)

    soft_calls = ()
    if enable_soft_calls:
        soft_calls = parse_soft_calls(
            parse_int_list(soft_call_years_text),
            parse_float_list(soft_call_prices_text),
            soft_call_trigger,
        )
    puts = ()
    if enable_puts:
        puts = parse_puts(parse_int_list(put_years_text), parse_float_list(put_prices_text))

    params = ConvertibleParams(
        evaluation_date=_ql_date(evaluation_date),
        spot=float(spot),
        volatility=float(volatility),
        dividend_yield=float(dividend_yield),
        risk_free_rate=float(risk_free_rate),
        credit_spread=float(credit_spread),
        coupon=float(coupon),
        frequency=frequency,
        tenor_years=int(tenor_years),
        conversion_ratio=conversion_ratio,
        redemption=float(redemption),
        settlement_days=int(settlement_days),
        american=american,
        tree_type=tree_type,
        time_steps=int(time_steps),
        soft_calls=soft_calls,
        puts=puts,
        enable_soft_calls=enable_soft_calls,
        enable_puts=enable_puts,
        dividend_amount=float(dividend_amount),
        dividend_period_months=int(dividend_period_months),
        enable_discrete_dividends=enable_discrete_dividends,
        bump_rel=float(bump_rel),
    )
    chart_cfg = {
        "spot_min": float(spot) * float(spot_pct_min) / 100.0,
        "spot_max": float(spot) * float(spot_pct_max) / 100.0,
        "n_points": int(n_points),
    }
    return params, chart_cfg, scheme, int(ayache_nodes)


def _line_chart(x, y, title: str, spot: float, y_title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=y_title))
    fig.add_vline(x=spot, line_width=1, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=title,
        xaxis_title="Stock price",
        yaxis_title=y_title,
        margin=dict(l=40, r=20, t=50, b=40),
        height=320,
    )
    return fig


def _render_total_default(nodes: int) -> None:
    """Table 1 total-default contract. The dashed line is the table spot S = 100."""
    spot_min, spot_max, n_points = 20.0, 200.0, 91
    try:
        with st.spinner("Pricing the Ayache total-default contract…"):
            greeks, grid = _cached_total_default(nodes, spot_min, spot_max, n_points)
    except Exception as exc:  # noqa: BLE001 — surface solver errors in the UI
        st.error(str(exc))
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Price at 100", f"{greeks['npv']:,.4f}")
    c2.metric("Delta", f"{greeks['delta']:,.4f}")
    c3.metric("Gamma", f"{greeks['gamma']:,.6f}")
    st.write(
        "Scheme **Ayache total default** · face 100 · conversion ratio 1 · "
        f"coupon 4 semiannual · r = 0.05 · hazard 0.02 · vol 20% · "
        f"**{nodes}** nodes · call 110 after year 2 · put 105 at year 3"
    )
    st.caption("Sidebar market and bond inputs are not used for this preset.")

    tab_price, tab_delta, tab_gamma = st.tabs(["Price", "Delta", "Gamma"])
    with tab_price:
        st.plotly_chart(
            _line_chart(grid["spot"], grid["npv"], "Total-default price vs spot", TABLE_SPOT, "Price"),
            use_container_width=True,
        )
    with tab_delta:
        st.plotly_chart(
            _line_chart(grid["spot"], grid["delta"], "Delta vs spot", TABLE_SPOT, "Delta"),
            use_container_width=True,
        )
    with tab_gamma:
        st.plotly_chart(
            _line_chart(grid["spot"], grid["gamma"], "Gamma vs spot", TABLE_SPOT, "Gamma"),
            use_container_width=True,
        )
    with st.expander("Grid data"):
        st.dataframe(grid, use_container_width=True)


def main() -> None:
    st.title("Convertible Bond Pricer")
    st.caption(
        "Fixed-coupon convertible with soft call / put. Binomial TF is QuantLib's tree. "
        "Crank–Nicolson TF is the Ayache–Forsyth–Vetzal theta scheme on the same split. "
        "Ayache total default is the Table 1 hedge model with the short rate replaced by r + p. "
        "Binomial Greeks are bump-and-reprice; the finite-difference delta and gamma are grid derivatives."
    )

    params, chart_cfg, scheme, ayache_nodes = _build_params_from_sidebar()

    if scheme == "Ayache total default":
        _render_total_default(ayache_nodes)
        return

    try:
        with st.spinner("Pricing…"):
            key = _params_to_tuple(params)
            if scheme == "Crank-Nicolson TF":
                greeks, grid = _cached_crank_nicolson(
                    key,
                    chart_cfg["spot_min"],
                    chart_cfg["spot_max"],
                    chart_cfg["n_points"],
                )
            else:
                greeks, grid = _cached_binomial(
                    key,
                    chart_cfg["spot_min"],
                    chart_cfg["spot_max"],
                    chart_cfg["n_points"],
                )
    except Exception as exc:  # noqa: BLE001 — surface QuantLib / input errors in UI
        st.error(str(exc))
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("NPV", f"{greeks['npv']:,.4f}")
    c2.metric("Delta", f"{greeks['delta']:,.4f}")
    c3.metric("Gamma", f"{greeks['gamma']:,.6f}")
    c4.metric("Vega (per vol pt)", f"{greeks['vega']:,.4f}")
    c5.metric("Credit spread 01", f"{greeks['credit_spread_01']:,.6f}")

    ratio = (
        params.redemption / params.spot
        if params.conversion_ratio is None
        else params.conversion_ratio
    )
    engine_label = (
        f"Tree **{params.tree_type}**"
        if scheme == "Binomial TF"
        else "Crank–Nicolson"
    )
    st.write(
        f"Scheme **{scheme}** · "
        f"Conversion ratio **{ratio:.6f}** · "
        f"{engine_label} · "
        f"Steps **{params.time_steps}** · "
        f"{'American' if params.american else 'European'} conversion"
    )

    tab_price, tab_delta, tab_gamma = st.tabs(["Price", "Delta", "Gamma"])
    with tab_price:
        st.plotly_chart(
            _line_chart(grid["spot"], grid["npv"], "Convertible price vs spot", params.spot, "NPV"),
            use_container_width=True,
        )
    with tab_delta:
        st.plotly_chart(
            _line_chart(grid["spot"], grid["delta"], "Delta vs spot", params.spot, "Delta"),
            use_container_width=True,
        )
    with tab_gamma:
        st.plotly_chart(
            _line_chart(grid["spot"], grid["gamma"], "Gamma vs spot", params.spot, "Gamma"),
            use_container_width=True,
        )

    with st.expander("Grid data"):
        st.dataframe(grid, use_container_width=True)

    st.info(
        "Defaults follow `Examples/ConvertibleBonds`. Soft-call / put indices are "
        "positions on the bond coupon schedule (for annual frequency, index ≈ years from issue)."
    )


if __name__ == "__main__":
    main()
