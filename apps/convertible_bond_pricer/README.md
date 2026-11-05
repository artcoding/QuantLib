# Convertible Bond Pricer

Streamlit desktop app for pricing a **fixed-coupon convertible bond** with soft call / put features, using QuantLib’s Tsiveriotis–Fernandes binomial engine.

## Features

- Market and bond parameter controls (spot, vol, rates, credit spread, coupon, conversion ratio)
- Soft call and put schedules (aligned with the QuantLib C++ `Examples/ConvertibleBonds` setup)
- NPV plus Greeks (delta, gamma, vega, credit-spread sensitivity)
- Plots of price, delta, and gamma against stock price
- Scheme choice: QuantLib binomial Tsiveriotis–Fernandes, or a Crank–Nicolson finite-difference scheme for the same cash/equity split (Ayache, Forsyth, and Vetzal, Appendix A.1)

Greeks are **not** produced by the library engine; they are central finite differences on reprice.

## Requirements

A working [QuantLib-Python](https://pypi.org/project/QuantLib/) wheel for your platform (`pip install QuantLib`).

## Install

```bash
cd apps/convertible_bond_pricer
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

On Windows, if `streamlit` is not on `PATH`:

```bash
py -3 -m streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Notes

- Default tree is Cox–Ross–Rubinstein with 201 steps (interactive). Raise steps for closer agreement with the C++ example (801).
- Crank–Nicolson prices the same fixed coupon, soft call, put, and escrowed dividends. On the example defaults at 201 time steps its NPV is within a few tenths of a point of the binomial engine.
- Spot grids reprice the bond at each point; gamma needs three NPVs per grid point, so denser grids or more steps are slower.
- This app is standalone and is not part of the QuantLib C++ build or CI.
