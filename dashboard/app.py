
"""Streamlit dashboard for StockSense AI."""

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv(
    "STOCKSENSE_API_URL",
    "http://127.0.0.1:8000",
)

st.set_page_config(
    page_title="StockSense AI",
    page_icon="📈",
    layout="wide",
)

st.title("📈 StockSense AI")
st.caption("Experimental NIFTY 50 forecasting and analytics dashboard")

st.warning(
    "Experimental research project. Models have not demonstrated "
    "reliable predictive performance on the held-out test period. "
    "Not investment advice."
)


@st.cache_data(ttl=60)
def fetch_api_data(endpoint):
    """Fetch and cache data from the FastAPI backend."""
    response = requests.get(
        f"{API_URL}/{endpoint}",
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def format_percent(value):
    return f"{value:+.3f}%"


# Sidebar
st.sidebar.header("Dashboard Controls")
st.sidebar.write("Backend API")
st.sidebar.code(API_URL)

if st.sidebar.button("🔄 Refresh dashboard", width="stretch"):
    st.cache_data.clear()
    st.rerun()


# Fetch backend data
try:
    health = fetch_api_data("health")
    prediction = fetch_api_data("predict/latest")
    features = fetch_api_data("features/latest")
    model_info = fetch_api_data("model-info")

except requests.RequestException as exc:
    st.error(f"Unable to connect to the StockSense API: {exc}")
    st.info(
        "Make sure the FastAPI server is running on "
        "http://127.0.0.1:8000."
    )
    st.stop()


# API status
st.subheader("System Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("API Status", health["status"].upper())

with col2:
    st.metric(
        "Loaded Classifiers",
        len(health["models_loaded"]["classifiers"]),
    )

with col3:
    st.metric(
        "Loaded Regressors",
        len(health["models_loaded"]["regressors"]),
    )


# Market overview
st.divider()
st.subheader("Market Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Index",
        prediction["symbol"],
    )

with col2:
    st.metric(
        "Latest Close",
        f"{prediction['latest_close']:,.2f}",
    )

with col3:
    st.metric(
        "Market Data Date",
        prediction["market_data_date"],
    )

st.caption(
    f"Forecast horizon: {prediction['forecast_horizon']}"
)

# Data freshness
st.info(
    f"📅 Latest available market data: "
    f"{prediction['market_data_date']} | "
    "Source: Local historical CSV"
)

st.caption(
    "This is the latest date in the available local dataset, "
    "not a live market feed."
)

# Classification predictions
st.divider()
st.subheader("Market Direction Predictions")

classification = prediction["classification"]

classification_rows = []

classification_rows.extend(
    {
        "Model": model,
        "Predicted Direction": result["predicted_direction"],
        "Probability Up (%)": round(result["probability_up"] * 100, 2),
        "Probability Down (%)": round(result["probability_down"] * 100, 2),
    }
    for model, result in classification.items()
)
classification_df = pd.DataFrame(classification_rows)

# Prediction consensus
st.subheader("Model Consensus")

up_votes = sum(
    result["predicted_direction"] == "Up"
    for result in classification.values()
)

down_votes = len(classification) - up_votes
total_models = len(classification)

consensus_col1, consensus_col2, consensus_col3 = st.columns(3)

with consensus_col1:
    st.metric(
        "Models Predicting Up",
        f"{up_votes}/{total_models}",
    )

with consensus_col2:
    st.metric(
        "Models Predicting Down",
        f"{down_votes}/{total_models}",
    )

with consensus_col3:
    st.metric(
        "Model Agreement",
        f"{max(up_votes, down_votes)}/{total_models}",
    )

if up_votes > down_votes:
    st.success(f"{up_votes} of {total_models} models predict Up.")
elif down_votes > up_votes:
    st.info(f"{down_votes} of {total_models} models predict Down.")
else:
    st.warning("The models are evenly split.")

st.caption(
    "Consensus represents agreement between model outputs, "
    "not prediction accuracy or a measure of future market certainty."
)

st.caption(
    "Probabilities are model outputs, not calibrated guarantees "
    "of future market direction."
)

# Interactive classification probability chart
probability_chart = classification_df.melt(
    id_vars="Model",
    value_vars=[
        "Probability Up (%)",
        "Probability Down (%)",
    ],
    var_name="Direction",
    value_name="Probability",
)

probability_fig = px.bar(
    probability_chart,
    x="Model",
    y="Probability",
    color="Direction",
    barmode="group",
    text_auto=".2f", # pyright: ignore[reportArgumentType]
    range_y=[0, 100],
    labels={
        "Probability": "Model probability (%)",
        "Model": "Classification model",
    },
    title="Up vs Down Probability by Model",
    color_discrete_map={
        "Probability Up (%)": "#16A34A",
        "Probability Down (%)": "#DC2626",
    },
)

probability_fig.update_layout(
    yaxis_ticksuffix="%",
    yaxis_dtick=20,
    legend_title_text="Predicted direction",
)

st.plotly_chart(
    probability_fig,
    width="stretch",
)


# Regression predictions
st.divider()
st.subheader("Next-Session Return Forecasts")

regression = prediction["regression"]

regression_rows = []

regression_rows.extend(
    {
        "Model": model,
        "Predicted Return (%)": round(result["predicted_return_percent"], 4),
    }
    for model, result in regression.items()
)
regression_df = pd.DataFrame(regression_rows)

st.dataframe(
    regression_df,
    width="stretch",
    hide_index=True,
)

# Interactive regression forecast chart
regression_df["Forecast Type"] = regression_df[
    "Predicted Return (%)"
].apply(
    lambda value: "Positive" if value >= 0 else "Negative"
)

regression_fig = px.bar(
    regression_df,
    x="Model",
    y="Predicted Return (%)",
    color="Forecast Type",
    text="Predicted Return (%)",
    labels={
        "Predicted Return (%)": "Predicted return (%)",
        "Model": "Regression model",
    },
    title="Predicted Next-Session Returns",
    color_discrete_map={
        "Positive": "#16A34A",
        "Negative": "#DC2626",
    },
)

regression_fig.update_traces(
    texttemplate="%{y:+.4f}%",
    textposition="outside",
)

regression_fig.update_layout(
    yaxis_ticksuffix="%",
    yaxis_zeroline=True,
    yaxis_zerolinewidth=2,
    showlegend=False,
)

st.plotly_chart(
    regression_fig,
    width="stretch",
)


# Latest technical features
st.divider()
st.subheader("Latest Technical Features")

feature_values = features["features"]

feature_df = pd.DataFrame(
    [
        {"Feature": name, "Value": value}
        for name, value in feature_values.items()
    ]
)

with st.expander("View all 24 model features"):
    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True,
    )


# Model information
st.divider()
st.subheader("Model Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Classification Models**")
    for model in model_info["classifiers"]:
        st.write(f"- {model}")

with col2:
    st.markdown("**Regression Models**")
    for model in model_info["regressors"]:
        st.write(f"- {model}")

st.caption(
    f"Model status: {model_info['model_status']} | "
    f"Prediction status: {model_info['prediction_status']}"
)

st.divider()
st.caption(
    "StockSense AI | Data source: local historical NIFTY 50 CSV | "
    "Predictions are experimental and for educational purposes only."
)
