# -*- coding: utf-8 -*-
"""
From Black Box to Glass Box – Explainable PMS Experiment

AI selects 1 stock per sector (5 total).
User can override sector choices; weights fixed at 20% each.
Black-box vs XAI trust experiment, with 3-feature visual explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta, datetime
import uuid
import altair as alt

# ---------- CONFIG ----------

# 10 stocks total – 2 per sector
SECTOR_MAP = {
    "US Tech": ["AAPL", "MSFT"],
    "Auto / EV": ["TSLA", "VOLCAR-B.ST"],
    "Healthcare / Pharma": ["AZN.ST", "NOVO-B.CO"],
    "Banking": ["DANSKE.CO", "NDA-DK.CO"],
    "Energy / Industrials": ["VWS.CO", "MAERSK-B.CO"],
}

# Friendly names for tickers
STOCK_NAME_MAP = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "TSLA": "Tesla Inc.",
    "VOLCAR-B.ST": "Volvo Car AB (B)",
    "AZN.ST": "AstraZeneca PLC (SE)",
    "NOVO-B.CO": "Novo Nordisk A/S (B)",
    "DANSKE.CO": "Danske Bank A/S",
    "NDA-DK.CO": "Nordea Bank Abp",
    "VWS.CO": "Vestas Wind Systems A/S",
    "MAERSK-B.CO": "A.P. Møller – Mærsk A/S (B)",
}

# Flatten all tickers
ALL_TICKERS = [t for pair in SECTOR_MAP.values() for t in pair]

YEARS_HISTORY = 5
TRADING_DAYS = 252
MC_PATHS = 3000
FUTURE_YEARS = 5

# Experiment metadata
EXPERIMENT_NAME = "xai_pms_trust_v2"
LOG_FILE = "experiment_logs.csv"

st.set_page_config(page_title="Explainable Portfolio Playground", layout="wide")


# ---------- EXPERIMENT HELPERS ----------

def get_session_id():
    """Give each browser session a unique ID."""
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


def log_response(row_dict):
    """Append one response as a row into experiment_logs.csv"""
    row_dict["timestamp"] = datetime.utcnow().isoformat()
    df_row = pd.DataFrame([row_dict])

    try:
        existing = pd.read_csv(LOG_FILE)
        df_out = pd.concat([existing, df_row], ignore_index=True)
    except FileNotFoundError:
        df_out = df_row

    df_out.to_csv(LOG_FILE, index=False)


# ---------- CORE PORTFOLIO LOGIC ----------

@st.cache_data(show_spinner=False)
def load_prices(tickers, years=YEARS_HISTORY):
    """Download and clean historical prices for given tickers."""
    end = date.today()
    start = end - timedelta(days=int(years * 365.25))
    data = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, group_by="ticker", threads=True
    )
    frames = []
    for t in tickers:
        try:
            if (t, "Close") in data.columns:
                s = data[(t, "Close")].rename(t)
            elif (t, "Adj Close") in data.columns:
                s = data[(t, "Adj Close")].rename(t)
            else:
                continue
            frames.append(s)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    prices = pd.concat(frames, axis=1)
    prices = prices.asfreq("B").ffill()
    return prices.dropna(how="all", axis=1)


def annualize_stats(rets):
    """Compute annualised mean return, vol and covariance from daily returns."""
    mu_daily = rets.mean()
    cov_daily = rets.cov()
    mu_ann = (1 + mu_daily).pow(TRADING_DAYS) - 1
    vol_ann = rets.std() * np.sqrt(TRADING_DAYS)
    return mu_ann, vol_ann, cov_daily * TRADING_DAYS


def bootstrap_paths(rets, weights, years=FUTURE_YEARS, paths=MC_PATHS):
    """Bootstrap daily returns to simulate many future paths."""
    np.random.seed(42)
    n = rets.shape[0]
    horizon = int(TRADING_DAYS * years)
    port_daily = rets.dot(weights).values
    draws = np.random.choice(np.arange(n), size=(paths, horizon), replace=True)
    sim = (port_daily[draws] + 1.0).cumprod(axis=1)
    return sim


def explainability_table(prices, rets, weights):
    """
    Build a plain-English explanation table for chosen stocks.
    Only show stocks with non-zero weights.
    Features:
    - 12m Return (%)
    - 60d Volatility (annualised, %)
    - 60d Avg Correlation
    """
    weights = pd.Series(weights)
    weights = weights.reindex(prices.columns).fillna(0)

    last = prices.index[-1]
    one_year = prices.index[-TRADING_DAYS:]
    sixty = prices.index[-60:]

    # 12m return
    r12 = prices.loc[last] / prices.loc[one_year[0]] - 1
    # 60d vol (annualised)
    v60 = rets.loc[sixty].std() * np.sqrt(TRADING_DAYS)
    # 60d avg correlation
    corr_60 = rets.loc[sixty].corr()
    avg_corr = corr_60.mean()

    df = pd.DataFrame({
        "12m Return (%)": (r12 * 100).round(2),
        "Ann. Vol (60d, %)": (v60 * 100).round(2),
        "Avg Correlation (60d)": avg_corr.round(2),
        "Weight (%)": (weights * 100).round(1),
    })

    # Only keep chosen stocks (weight > 0)
    df = df[df["Weight (%)"] > 0].sort_values("Weight (%)", ascending=False)

    def rationale(row):
        msgs = []
        if row["12m Return (%)"] > df["12m Return (%)"].median():
            msgs.append("strong recent return")
        if row["Ann. Vol (60d, %)"] < df["Ann. Vol (60d, %)"].median():
            msgs.append("lower risk")
        if row["Avg Correlation (60d)"] < df["Avg Correlation (60d)"].median():
            msgs.append("adds diversification")
        return ", ".join(msgs) if msgs else "balanced profile"

    df["Why chosen (plain English)"] = df.apply(rationale, axis=1)
    return df


# ---------- INTRO / WELCOME SCREEN ----------

if "started_experiment" not in st.session_state:
    st.session_state["started_experiment"] = False

if not st.session_state["started_experiment"]:
    st.title("Explainable Portfolio Playground")
    st.subheader("Welcome 👋")

    st.write(
        "In this study, an AI system will help you build a stock portfolio. "
        "Sometimes it will behave like a **black box**, and sometimes it will "
        "show **explanations** for its choices."
    )

    st.markdown("### What you will do")
    st.markdown(
        """
        1. Tell us your **investment profile** (amount, return target, risk appetite).  
        2. The AI will pick **one stock in each sector** (5 in total).  
        3. You can **accept or change** each sector choice.  
        4. You’ll see **5-year outcome simulations** based on your portfolio.  
        5. Finally, you’ll rate **how much you trust** this AI-assisted portfolio.
        """
    )

    with st.expander("What is this experiment about?"):
        st.write(
            "We are interested in how people **trust** AI systems in financial decision-making. "
            "By comparing a black-box AI with an explainable AI, we want to understand "
            "whether explanations change your confidence in the recommendation."
        )

    if st.button("Start experiment"):
        st.session_state["started_experiment"] = True
        st.rerun()

    st.stop()  # Do not run the rest of the app until user starts


# ---------- MAIN UI STARTS HERE ----------

st.title("Explainable Portfolio Playground")
st.caption(
    "AI selects one stock per sector (5 total), with equal 20% weights. "
    "You can override its choices and see how it affects risk and return. "
    "Use the toggle to compare black-box vs explainable AI."
)

# For your experiment: optional participant ID
participant_id = st.text_input("Participant ID or initials (optional):", "")

# --- Investor profile inputs ---
st.subheader("0) Your investor profile")

amount = st.number_input(
    "Investment amount (DKK):",
    min_value=10000,
    value=100000,
    step=10000
)

target_return = st.slider(
    "Target annual return (%)",
    min_value=10.0,
    max_value=22.0,
    value=14.0,
    step=0.5
)

risk_profile = st.radio(
    "Risk appetite:",
    ["Conservative", "Balanced", "Aggressive"],
    index=1
)

st.caption(
    f"You aim for about {target_return:.1f}% annual return with a "
    f"{risk_profile.lower()} risk profile on an investment of {amount:,.0f} DKK."
)

# --- Explanation mode toggle ---
st.subheader("1) Explanation mode")

mode = st.radio(
    "How should the AI behave?",
    [
        "Black-box view (no explanations)",
        "Explainable AI (show reasons for each chosen stock)"
    ],
    index=0,
    help="In black-box mode, you see the outcome but not the reasoning. "
         "In Explainable AI mode, you also see why each stock was selected."
)

st.divider()

# Load data for all 10 tickers
prices = load_prices(ALL_TICKERS, YEARS_HISTORY)
if prices.empty:
    st.error("Could not load price data. Check tickers or try again.")
    st.stop()

rets = prices.pct_change().dropna()
mu_ann, vol_ann, cov_ann = annualize_stats(rets)

# Pre-compute correlation info for diversification
sixty = rets.index[-60:]
corr_60 = rets.loc[sixty].corr()
avg_corr = corr_60.mean()

# ---------- 2) SECTOR-WISE CHOICES WITH GLOWING AI SUGGESTION ----------

# ---------- 2) SECTOR-WISE CHOICES (Soft Highlight Style) ----------

st.subheader("2) Sector-wise choices")

selected_tickers = []
sector_choices = {}

sectors = list(SECTOR_MAP.keys())

# Create columns (3 in first row, 2 in second)
row1_cols = st.columns(3)
for col, sector in zip(row1_cols, sectors[:3]):
    with col:
        tickers = SECTOR_MAP[sector]
        mu = mu_ann.loc[tickers]
        vol = vol_ann.loc[tickers]
        corr = avg_corr.loc[tickers]

        df_sec = pd.DataFrame({"mu": mu, "vol": vol, "corr": corr})
        df_norm = (df_sec - df_sec.min()) / (df_sec.max() - df_sec.min() + 1e-9)

        # Risk-based scoring
        if risk_profile == "Aggressive":
            score = 0.7 * df_norm["mu"] - 0.2 * df_norm["vol"] - 0.1 * df_norm["corr"]
        elif risk_profile == "Conservative":
            score = 0.2 * df_norm["mu"] - 0.5 * df_norm["vol"] - 0.3 * df_norm["corr"]
        else:
            score = 0.5 * df_norm["mu"] - 0.3 * df_norm["vol"] - 0.2 * df_norm["corr"]

        df_sec["score"] = score
        ai_pick = df_sec["score"].idxmax()

        st.markdown(f"**{sector}**")
        names_str = " • ".join(
            f"{t} – {STOCK_NAME_MAP.get(t, '')}" for t in tickers
        )
        st.caption(f"Options: {names_str}")

        # ✨ Soft highlight badge for AI suggestion
        st.markdown(
            f"""
            <p style="
                background-color:#EAF9F2;
                color:#00875A;
                font-weight:600;
                display:inline-block;
                padding:4px 10px;
                border-radius:6px;
                margin-top:4px;
                margin-bottom:8px;">
                🤖 AI Suggestion: {ai_pick}
            </p>
            """,
            unsafe_allow_html=True
        )

        choice = st.radio(
            f"Choose 1 stock for {sector}:",
            tickers,
            index=tickers.index(ai_pick),
            key=f"radio_{sector}"
        )

        sector_choices[sector] = choice
        selected_tickers.append(choice)

# Second row (2 sectors)
row2_cols = st.columns(2)
for col, sector in zip(row2_cols, sectors[3:]):
    with col:
        tickers = SECTOR_MAP[sector]
        mu = mu_ann.loc[tickers]
        vol = vol_ann.loc[tickers]
        corr = avg_corr.loc[tickers]

        df_sec = pd.DataFrame({"mu": mu, "vol": vol, "corr": corr})
        df_norm = (df_sec - df_sec.min()) / (df_sec.max() - df_sec.min() + 1e-9)

        if risk_profile == "Aggressive":
            score = 0.7 * df_norm["mu"] - 0.2 * df_norm["vol"] - 0.1 * df_norm["corr"]
        elif risk_profile == "Conservative":
            score = 0.2 * df_norm["mu"] - 0.5 * df_norm["vol"] - 0.3 * df_norm["corr"]
        else:
            score = 0.5 * df_norm["mu"] - 0.3 * df_norm["vol"] - 0.2 * df_norm["corr"]

        df_sec["score"] = score
        ai_pick = df_sec["score"].idxmax()

        st.markdown(f"**{sector}**")
        names_str = " • ".join(
            f"{t} – {STOCK_NAME_MAP.get(t, '')}" for t in tickers
        )
        st.caption(f"Options: {names_str}")

        # ✨ Soft highlight badge for AI suggestion
        st.markdown(
            f"""
            <p style="
                background-color:#EAF9F2;
                color:#00875A;
                font-weight:600;
                display:inline-block;
                padding:4px 10px;
                border-radius:6px;
                margin-top:4px;
                margin-bottom:8px;">
                🤖 AI Suggestion: {ai_pick}
            </p>
            """,
            unsafe_allow_html=True
        )

        choice = st.radio(
            f"Choose 1 stock for {sector}:",
            tickers,
            index=tickers.index(ai_pick),
            key=f"radio_{sector}"
        )

        sector_choices[sector] = choice
        selected_tickers.append(choice)

st.caption(
    "The final portfolio will include one stock from each sector, "
    "with equal 20% weights."
)


st.caption(
    "The final portfolio will include one stock from each sector, "
    "with equal 20% weights."
)

# ---------- PORTFOLIO CALCULATIONS ----------

weights_series = pd.Series(0.0, index=ALL_TICKERS)
n_sel = len(selected_tickers)
if n_sel == 0:
    st.error("No stocks selected. Something went wrong.")
    st.stop()

equal_weight = 1.0 / n_sel
weights_series[selected_tickers] = equal_weight

mu_sel = mu_ann.loc[selected_tickers]
cov_sel = cov_ann.loc[selected_tickers, selected_tickers].values
w_sel = np.array([equal_weight] * n_sel)

port_mu = float(mu_sel.dot(w_sel))
port_vol = float(np.sqrt(w_sel @ cov_sel @ w_sel))

# ---------- 3 & 4: GROWTH CHART + OUTCOME TABLE ----------

st.subheader("3) Projected portfolio growth (5 years)")

sim = bootstrap_paths(rets[selected_tickers], pd.Series(w_sel, index=selected_tickers))
years_axis = np.arange(1, sim.shape[1] + 1) / TRADING_DAYS
median = np.median(sim, axis=0)
p05 = np.percentile(sim, 5, axis=0)
p95 = np.percentile(sim, 95, axis=0)

left, right = st.columns(2)
with left:
    df_proj = pd.DataFrame({
        "Year": years_axis,
        "P05": p05,
        "Median": median,
        "P95": p95
    })
    st.line_chart(df_proj.set_index("Year"))
    st.caption(
        "The band shows pessimistic (P05), typical (median), and optimistic (P95) "
        "paths based on Monte Carlo simulations."
    )

with right:
    st.subheader("4) Possible outcomes at Year 5")

    yr5 = sim[:, -1]

    median_mult = float(np.median(yr5))      # typical outcome
    p05_mult = float(np.percentile(yr5, 5))  # conservative / downside
    p95_mult = float(np.percentile(yr5, 95)) # optimistic / upside

    scenarios = [
        ("Conservative (5th percentile)", p05_mult),
        ("Typical (median)", median_mult),
        ("Optimistic (95th percentile)", p95_mult),
    ]

    rows = []
    for label, mult in scenarios:
        final_value = amount * mult
        total_return_pct = (mult - 1) * 100
        rows.append({
            "Scenario": label,
            "Total value after 5 years (DKK)": final_value,
            "Total return over 5 years (%)": total_return_pct,
        })

    stats_df = pd.DataFrame(rows)

    st.write(
        stats_df.style.format({
            "Total value after 5 years (DKK)": "{:,.0f}",
            "Total return over 5 years (%)": "{:.1f}",
        })
    )

    st.caption(
        "Scenarios are based on Monte Carlo simulations of your portfolio. "
        "Values show total change over 5 years, both in DKK and as a percentage."
    )

# Expected annual stats + comparison to target
st.markdown(
    f"**Expected annual return (model-based):** {port_mu*100:.2f}%  |  "
    f"**Expected annual volatility:** {port_vol*100:.2f}%"
)

delta_pp = port_mu * 100 - target_return
if delta_pp >= 0.2:
    st.success(
        f"This portfolio's expected return is about {delta_pp:.1f} percentage points "
        f"**above** your target of {target_return:.1f}% per year."
    )
elif delta_pp <= -0.2:
    st.warning(
        f"This portfolio's expected return is about {abs(delta_pp):.1f} percentage points "
        f"**below** your target of {target_return:.1f}% per year."
    )
else:
    st.info(
        f"This portfolio's expected return is very close to your target "
        f"of {target_return:.1f}% per year."
    )

st.divider()

# ---------- 5) FINAL PORTFOLIO SUMMARY ----------

st.subheader("5) Final 5-stock portfolio (equal weights)")

alloc_df = pd.DataFrame({
    "Sector": list(sector_choices.keys()),
    "Ticker": [sector_choices[s] for s in sector_choices],
    "Company": [STOCK_NAME_MAP.get(sector_choices[s], "") for s in sector_choices],
    "Weight (%)": [equal_weight * 100] * n_sel
})
st.dataframe(alloc_df, hide_index=True, use_container_width=True)

st.divider()

# ---------- 6) EXPLANATIONS OR BLACK-BOX MESSAGE ----------

st.subheader("6) Explanations for the chosen stocks")

if "Explainable" in mode:
    st.markdown("#### Why these 5 stocks? (simple, explainable factors)")
    exp_df = explainability_table(prices, rets, weights_series)
    # Add company names to explanation table for readability
    exp_df.insert(
        0, "Company",
        [STOCK_NAME_MAP.get(t, "") for t in exp_df.index]
    )
    st.dataframe(exp_df, use_container_width=True)

    st.caption(
        "Explanations are based on three features:\n"
        "- **12-month return** (higher is better for growth)\n"
        "- **60-day volatility** (lower is less risky)\n"
        "- **60-day average correlation** (lower gives more diversification)."
    )

    # --- Visual representation of each stock on the 3 core features ---
    st.markdown("#### Visual comparison of the chosen stocks")

    feat_df = exp_df[["12m Return (%)", "Ann. Vol (60d, %)", "Avg Correlation (60d)"]].copy()
    feat_df = feat_df.apply(pd.to_numeric, errors="coerce")
    feat_df = feat_df.rename(
        columns={
            "12m Return (%)": "Return",
            "Ann. Vol (60d, %)": "Volatility",
            "Avg Correlation (60d)": "Correlation",
        }
    )

    chart_df = feat_df.reset_index().rename(columns={"index": "Ticker"})

    # Create categories for color-coding (above/below median)
    ret_median = chart_df["Return"].median()
    vol_median = chart_df["Volatility"].median()
    corr_median = chart_df["Correlation"].median()

    chart_df["Return_cat"] = np.where(
        chart_df["Return"] >= ret_median, "Higher return", "Lower return"
    )
    chart_df["Vol_cat"] = np.where(
        chart_df["Volatility"] <= vol_median, "Lower risk", "Higher risk"
    )
    chart_df["Corr_cat"] = np.where(
        chart_df["Correlation"] <= corr_median, "Better diversification", "More correlated"
    )

    c1, c2, c3 = st.columns(3)

    tooltips = ["Ticker", "Return", "Volatility", "Correlation"]

    with c1:
        st.markdown("**Return (12m, %)**")
        chart_ret = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("Ticker:N", sort=None),
                y=alt.Y("Return:Q", title="Return (12m, %)"),
                color=alt.Color("Return_cat:N", title="Return vs median"),
                tooltip=tooltips,
            )
        )
        st.altair_chart(chart_ret, use_container_width=True)

    with c2:
        st.markdown("**Volatility (60d, %)**")
        chart_vol = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("Ticker:N", sort=None),
                y=alt.Y("Volatility:Q", title="Volatility (60d, %)"),
                color=alt.Color("Vol_cat:N", title="Risk level"),
                tooltip=tooltips,
            )
        )
        st.altair_chart(chart_vol, use_container_width=True)

    with c3:
        st.markdown("**Diversification (Avg correlation)**")
        chart_corr = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("Ticker:N", sort=None),
                y=alt.Y("Correlation:Q", title="Avg correlation (60d)"),
                color=alt.Color("Corr_cat:N", title="Diversification"),
                tooltip=tooltips,
            )
        )
        st.altair_chart(chart_corr, use_container_width=True)

    st.caption(
        "Colors show whether each stock is above/below the group on return, risk, and diversification. "
        "Hover over bars to see exact values."
    )

else:
    st.info(
        "Black-box mode: you see which stocks are in the portfolio and the overall performance, "
        "but not the detailed reasoning behind each choice."
    )

st.divider()

# ---------- 7) TRUST / FOLLOW-UP & LOGGING ----------

st.subheader("7) How much do you trust this AI-assisted portfolio?")

trust_score = st.slider(
    "Trust in this recommendation (0 = no trust, 10 = complete trust):",
    0, 10, 5
)
follow_score = st.slider(
    "How likely are you to follow this portfolio in real life?",
    0, 10, 5
)
comments = st.text_area("Any comments or thoughts about what you saw?", "")

if st.button("Submit your response"):
    session_id = get_session_id()
    condition = "Explainable" if "Explainable" in mode else "Black-box"

    row = {
        "experiment": EXPERIMENT_NAME,
        "session_id": session_id,
        "participant_id": participant_id,
        "condition": condition,
        "amount_dkk": amount,
        "target_return_pct": target_return,
        "risk_profile": risk_profile,
        "trust_score": trust_score,
        "follow_score": follow_score,
        "comments": comments,
        "expected_return": port_mu,
        "volatility": port_vol,
        "year5_median_growth_mult": median_mult,
        "year5_p05_mult": p05_mult,
        "year5_p95_mult": p95_mult,
    }

    # Top 3 positions for later analysis
    weights_nonzero = weights_series[weights_series > 0]
    top = weights_nonzero.sort_values(ascending=False).head(3)
    for i, (tkr, w) in enumerate(top.items(), start=1):
        row[f"top_{i}_asset"] = tkr
        row[f"top_{i}_weight"] = float(w)

    log_response(row)
    st.success("Thanks! Your response has been recorded.")
