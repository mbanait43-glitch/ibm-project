"""
app.py  –  E-Commerce Sales Prediction (single-file Streamlit app)
-------------------------------------------------------------------
All data processing, ML model logic, and the Streamlit UI live here.

Run:
    streamlit run app.py
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_BASE_DIR, "basket_details.csv")
_MODELS_DIR = os.path.join(_BASE_DIR, "models")
os.makedirs(_MODELS_DIR, exist_ok=True)

RF_PATH      = os.path.join(_MODELS_DIR, "rf_model.joblib")
XGB_PATH     = os.path.join(_MODELS_DIR, "xgb_model.joblib")
METRICS_PATH = os.path.join(_MODELS_DIR, "model_metrics.json")

# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "day_of_week", "month", "week_of_year", "day_of_month",
    "is_weekend", "lag_1", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_mean_14",
]
TARGET_COL = "total_sales"

# ===========================================================================
# DATA PROCESSING
# ===========================================================================

@st.cache_data(show_spinner=False)
def load_raw() -> pd.DataFrame:
    """Load the raw CSV with basket_date parsed."""
    return pd.read_csv(CSV_PATH, parse_dates=["basket_date"])


@st.cache_data(show_spinner=False)
def get_daily_sales(_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to daily totals and attach calendar + lag + rolling features.
    The leading underscore prevents Streamlit from hashing the DataFrame arg.
    """
    daily = (
        _df.groupby("basket_date")
        .agg(
            total_sales=("basket_count", "sum"),
            num_transactions=("basket_count", "count"),
            unique_customers=("customer_id", "nunique"),
            unique_products=("product_id", "nunique"),
        )
        .reset_index()
        .sort_values("basket_date")
    )

    daily["day_of_week"]  = daily["basket_date"].dt.dayofweek
    daily["month"]        = daily["basket_date"].dt.month
    daily["week_of_year"] = daily["basket_date"].dt.isocalendar().week.astype(int)
    daily["day_of_month"] = daily["basket_date"].dt.day
    daily["is_weekend"]   = daily["day_of_week"].isin([5, 6]).astype(int)

    daily["lag_1"]  = daily["total_sales"].shift(1)
    daily["lag_7"]  = daily["total_sales"].shift(7)
    daily["lag_14"] = daily["total_sales"].shift(14)

    daily["rolling_mean_7"]  = daily["total_sales"].shift(1).rolling(7,  min_periods=1).mean()
    daily["rolling_mean_14"] = daily["total_sales"].shift(1).rolling(14, min_periods=1).mean()

    return daily.reset_index(drop=True)


def get_summary_stats(df: pd.DataFrame) -> dict:
    daily = get_daily_sales(df)
    return {
        "total_rows":         int(len(df)),
        "total_sales":        int(df["basket_count"].sum()),
        "unique_customers":   int(df["customer_id"].nunique()),
        "unique_products":    int(df["product_id"].nunique()),
        "date_start":         str(df["basket_date"].min().date()),
        "date_end":           str(df["basket_date"].max().date()),
        "num_days":           int(daily["basket_date"].nunique()),
        "avg_daily_sales":    round(float(daily["total_sales"].mean()), 2),
        "max_daily_sales":    int(daily["total_sales"].max()),
        "min_daily_sales":    int(daily["total_sales"].min()),
        "avg_basket_size":    round(float(df["basket_count"].mean()), 2),
    }


def get_top_products(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    top = (
        df.groupby("product_id")
        .agg(
            total_sales=("basket_count", "sum"),
            num_transactions=("basket_count", "count"),
            unique_customers=("customer_id", "nunique"),
        )
        .reset_index()
        .sort_values("total_sales", ascending=False)
        .head(n)
    )
    top["product_id"] = top["product_id"].astype(str)
    return top


def get_customer_stats(df: pd.DataFrame) -> dict:
    cust = (
        df.groupby("customer_id")
        .agg(
            total_purchases=("basket_count", "sum"),
            num_transactions=("basket_count", "count"),
            unique_products=("product_id", "nunique"),
        )
        .reset_index()
    )

    top_customers = (
        cust.sort_values("total_purchases", ascending=False)
        .head(10)
        .assign(customer_id=lambda x: x["customer_id"].astype(str))
    )

    freq_series = cust["num_transactions"].value_counts().sort_index()
    freq_dist = pd.DataFrame({
        "transactions": freq_series.index,
        "count":        freq_series.values,
    }).head(20)

    return {
        "total_customers":                int(cust["customer_id"].nunique()),
        "avg_purchases_per_customer":     round(float(cust["total_purchases"].mean()), 2),
        "avg_transactions_per_customer":  round(float(cust["num_transactions"].mean()), 2),
        "top_customers":                  top_customers,
        "purchase_frequency_distribution": freq_dist,
    }


# ===========================================================================
# ML MODEL
# ===========================================================================

def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2":   round(float(r2_score(y_true, y_pred)), 4),
    }


def train_models(daily_df: pd.DataFrame) -> dict:
    """Train RF (+ XGBoost if available), save to disk, return metrics dict."""
    df = daily_df.dropna(subset=FEATURE_COLS).copy()
    X, y = df[FEATURE_COLS].values, df[TARGET_COL].values

    split = max(1, min(int(len(df) * 0.8), len(df) - 1))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=10,
        min_samples_leaf=1, random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_metrics = _calc_metrics(y_test, rf_pred)
    joblib.dump(rf, RF_PATH)

    # XGBoost
    if _XGBOOST_AVAILABLE:
        xgb = XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        xgb_metrics = _calc_metrics(y_test, xgb_pred)
        joblib.dump(xgb, XGB_PATH)
        ens_pred    = (rf_pred + xgb_pred) / 2.0
        ens_metrics = _calc_metrics(y_test, ens_pred)
    else:
        xgb_metrics = {"mae": None, "rmse": None, "r2": None}
        ens_metrics = rf_metrics

    importance = dict(sorted(
        {col: round(float(imp), 6) for col, imp in zip(FEATURE_COLS, rf.feature_importances_)}.items(),
        key=lambda kv: kv[1], reverse=True,
    ))

    result = {
        "random_forest":       rf_metrics,
        "xgboost":             xgb_metrics,
        "ensemble":            ens_metrics,
        "feature_importance":  importance,
        "train_size":          int(len(X_train)),
        "test_size":           int(len(X_test)),
        "xgboost_available":   _XGBOOST_AVAILABLE,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(result, f, indent=2)
    return result


def load_or_train(daily_df: pd.DataFrame) -> dict:
    """Return saved metrics if models exist, otherwise train first."""
    if os.path.exists(RF_PATH) and os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return train_models(daily_df)


def predict_future(daily_df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """Iterative one-step-ahead forecast for the next n_days days."""
    if not os.path.exists(RF_PATH):
        train_models(daily_df)

    rf  = joblib.load(RF_PATH)
    xgb = joblib.load(XGB_PATH) if (os.path.exists(XGB_PATH) and _XGBOOST_AVAILABLE) else None

    df        = daily_df.copy()
    last_date = df["basket_date"].max()
    records   = []

    for i in range(1, n_days + 1):
        nd  = last_date + pd.Timedelta(days=i)
        s   = df["total_sales"]
        row = {
            "day_of_week":      nd.dayofweek,
            "month":            nd.month,
            "week_of_year":     int(nd.isocalendar()[1]),
            "day_of_month":     nd.day,
            "is_weekend":       int(nd.dayofweek in [5, 6]),
            "lag_1":            float(s.iloc[-1])          if len(s) >= 1  else float(s.mean()),
            "lag_7":            float(s.iloc[-7])          if len(s) >= 7  else float(s.mean()),
            "lag_14":           float(s.iloc[-14])         if len(s) >= 14 else float(s.mean()),
            "rolling_mean_7":   float(s.iloc[-7:].mean()),
            "rolling_mean_14":  float(s.iloc[-14:].mean()),
        }
        X_pred  = np.array([[row[c] for c in FEATURE_COLS]])
        rf_pred = float(rf.predict(X_pred)[0])
        pred    = (rf_pred + float(xgb.predict(X_pred)[0])) / 2.0 if xgb else rf_pred
        pred    = max(0.0, pred)

        records.append({"date": nd.strftime("%Y-%m-%d"), "predicted_sales": round(pred, 2)})
        df = pd.concat([df, pd.DataFrame([{"basket_date": nd, "total_sales": pred, **row}])],
                       ignore_index=True)

    return pd.DataFrame(records)


# ===========================================================================
# STREAMLIT APP
# ===========================================================================

st.set_page_config(
    page_title="E-Commerce Sales Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data once ──────────────────────────────────────────────────────────
raw_df   = load_raw()
daily_df = get_daily_sales(raw_df)

# ── Ensure models are trained (runs once, results cached to disk) ────────────
with st.spinner("Loading / training models … (first run only)"):
    metrics = load_or_train(daily_df)

# ── Sidebar navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛒 Sales Dashboard")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "📊 Dashboard",
            "🔮 Sales Prediction",
            "📦 Product Analysis",
            "👤 Customer Analysis",
            "🤖 Model Performance",
        ],
    )
    st.markdown("---")
    st.caption(f"Dataset: `basket_details.csv`")
    st.caption(f"Rows: **{len(raw_df):,}**")

# ===========================================================================
# PAGE 1 – DASHBOARD
# ===========================================================================

if page == "📊 Dashboard":
    st.title("📊 Sales Dashboard")
    st.markdown("Overview of the e-commerce basket dataset.")

    s = get_summary_stats(raw_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales (units)",  f"{s['total_sales']:,}")
    c2.metric("Unique Customers",     f"{s['unique_customers']:,}")
    c3.metric("Unique Products",      f"{s['unique_products']:,}")
    c4.metric("Avg Basket Size",      f"{s['avg_basket_size']:.2f}")

    st.markdown("---")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Transactions",   f"{s['total_rows']:,}")
    c6.metric("Days in Dataset",      f"{s['num_days']}")
    c7.metric("Avg Daily Sales",      f"{s['avg_daily_sales']:,.1f}")
    c8.metric("Date Range",           f"{s['date_start']} → {s['date_end']}")

    st.markdown("---")
    st.subheader("Daily Sales Trend")

    trend_df = daily_df[["basket_date", "total_sales"]].copy()
    fig = px.line(
        trend_df, x="basket_date", y="total_sales",
        markers=True,
        title="Total Daily Basket Count",
        labels={"basket_date": "Date", "total_sales": "Total Sales (basket count)"},
        template="plotly_white",
    )
    fig.update_traces(line_color="#3b82d4", marker_color="#3b82d4")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sales by Day of Week")
    dow_df = trend_df.copy()
    dow_df["day_name"] = dow_df["basket_date"].dt.day_name()
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow_agg = (
        dow_df.groupby("day_name")["total_sales"].mean()
        .reindex(dow_order).reset_index()
        .rename(columns={"total_sales": "avg_sales"})
    )
    fig_dow = px.bar(
        dow_agg, x="day_name", y="avg_sales",
        title="Average Sales by Day of Week",
        labels={"day_name": "Day", "avg_sales": "Avg Sales"},
        color="avg_sales", color_continuous_scale="Blues",
        template="plotly_white",
    )
    fig_dow.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_dow, use_container_width=True)

# ===========================================================================
# PAGE 2 – SALES PREDICTION
# ===========================================================================

elif page == "🔮 Sales Prediction":
    st.title("🔮 Sales Prediction")
    st.markdown("Forecast future daily sales using a **Random Forest + XGBoost ensemble** model.")

    col_ctrl, _ = st.columns([1, 2])
    with col_ctrl:
        horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=30, value=14, step=1)

    with st.spinner(f"Generating {horizon}-day forecast …"):
        pred_df = predict_future(daily_df, horizon)

    pred_df["date"] = pd.to_datetime(pred_df["date"])

    # Combined historical + forecast chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_df["basket_date"], y=daily_df["total_sales"],
        mode="lines+markers", name="Historical",
        line=dict(color="#3b82d4"), marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=pred_df["date"], y=pred_df["predicted_sales"],
        mode="lines+markers", name="Forecast",
        line=dict(color="#f97316", dash="dash"),
        marker=dict(size=6, symbol="diamond"),
    ))
    fig.update_layout(
        title=f"{horizon}-Day Sales Forecast",
        xaxis_title="Date", yaxis_title="Sales (basket count)",
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Table")
    display_df = pred_df.copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    display_df.columns = ["Date", "Predicted Sales"]
    display_df["Predicted Sales"] = display_df["Predicted Sales"].round(0).astype(int)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    ca, cb, cc = st.columns(3)
    ca.metric("Forecast Total",    f"{pred_df['predicted_sales'].sum():,.0f}")
    cb.metric("Forecast Avg/day",  f"{pred_df['predicted_sales'].mean():,.1f}")
    cc.metric("Forecast Peak",     f"{pred_df['predicted_sales'].max():,.0f}")

# ===========================================================================
# PAGE 3 – PRODUCT ANALYSIS
# ===========================================================================

elif page == "📦 Product Analysis":
    st.title("📦 Product Analysis")

    col_ctrl, _ = st.columns([1, 3])
    with col_ctrl:
        top_n = st.selectbox("Show top N products", [5, 10, 15, 20], index=1)

    prod_df = get_top_products(raw_df, n=top_n)

    fig = px.bar(
        prod_df.sort_values("total_sales"),
        x="total_sales", y="product_id", orientation="h",
        title=f"Top {top_n} Products by Total Sales",
        labels={"total_sales": "Total Sales", "product_id": "Product ID"},
        color="total_sales", color_continuous_scale="Blues",
        template="plotly_white",
    )
    fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Unique Customers per Product")
        fig2 = px.bar(
            prod_df.sort_values("unique_customers", ascending=False),
            x="product_id", y="unique_customers",
            title="Unique Customers per Top Product",
            labels={"product_id": "Product ID", "unique_customers": "Unique Customers"},
            color="unique_customers", color_continuous_scale="Purples",
            template="plotly_white",
        )
        fig2.update_layout(coloraxis_showscale=False, xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Transactions per Product")
        fig3 = px.bar(
            prod_df.sort_values("num_transactions", ascending=False),
            x="product_id", y="num_transactions",
            title="Number of Transactions per Top Product",
            labels={"product_id": "Product ID", "num_transactions": "Transactions"},
            color="num_transactions", color_continuous_scale="Greens",
            template="plotly_white",
        )
        fig3.update_layout(coloraxis_showscale=False, xaxis_tickangle=-45)
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Product Details Table")
    disp = prod_df[["product_id","total_sales","num_transactions","unique_customers"]].copy()
    disp.columns = ["Product ID","Total Sales","Transactions","Unique Customers"]
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ===========================================================================
# PAGE 4 – CUSTOMER ANALYSIS
# ===========================================================================

elif page == "👤 Customer Analysis":
    st.title("👤 Customer Analysis")

    cust = get_customer_stats(raw_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers",             f"{cust['total_customers']:,}")
    c2.metric("Avg Purchases / Customer",    f"{cust['avg_purchases_per_customer']:.2f}")
    c3.metric("Avg Transactions / Customer", f"{cust['avg_transactions_per_customer']:.2f}")

    st.markdown("---")
    st.subheader("Top 10 Customers by Total Purchases")

    top_df = cust["top_customers"]
    fig = px.bar(
        top_df.sort_values("total_purchases"),
        x="total_purchases", y="customer_id", orientation="h",
        title="Top 10 Customers – Total Purchases",
        labels={"total_purchases": "Total Purchases", "customer_id": "Customer ID"},
        color="total_purchases", color_continuous_scale="Blues",
        template="plotly_white",
    )
    fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Unique Products per Customer (Top 10)")
        fig2 = px.bar(
            top_df.sort_values("unique_products", ascending=False),
            x="customer_id", y="unique_products",
            title="Product Variety – Top 10 Customers",
            labels={"customer_id": "Customer ID", "unique_products": "Unique Products"},
            color="unique_products", color_continuous_scale="Oranges",
            template="plotly_white",
        )
        fig2.update_layout(coloraxis_showscale=False, xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Purchase Frequency Distribution")
        freq_df = cust["purchase_frequency_distribution"]
        fig3 = px.bar(
            freq_df, x="transactions", y="count",
            title="How Many Transactions Do Customers Make?",
            labels={"transactions": "Number of Transactions", "count": "Number of Customers"},
            color="count", color_continuous_scale="Greens",
            template="plotly_white",
        )
        fig3.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Top Customers Table")
    disp = top_df[["customer_id","total_purchases","num_transactions","unique_products"]].copy()
    disp.columns = ["Customer ID","Total Purchases","Transactions","Unique Products"]
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ===========================================================================
# PAGE 5 – MODEL PERFORMANCE
# ===========================================================================

elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")
    st.markdown("Metrics for the **Random Forest + XGBoost ensemble** trained on the basket dataset.")

    rows = []
    model_map = {
        "random_forest": "Random Forest",
        "xgboost":       "XGBoost",
        "ensemble":      "Ensemble (avg)",
    }
    for key, label in model_map.items():
        m = metrics.get(key, {})
        if m.get("mae") is not None:
            rows.append({"Model": label, "MAE": m["mae"], "RMSE": m["rmse"], "R²": m["r2"]})

    metrics_df = pd.DataFrame(rows)

    if len(rows) > 0:
        best = metrics_df.loc[metrics_df["MAE"].idxmin()]
        c1, c2, c3 = st.columns(3)
        c1.metric("Best MAE",  f"{best['MAE']:.2f}",    delta=best["Model"], delta_color="off")
        c2.metric("Best RMSE", f"{best['RMSE']:.2f}",   delta=best["Model"], delta_color="off")
        c3.metric("Best R²",   f"{best['R²']:.4f}",     delta=best["Model"], delta_color="off")

    st.markdown("---")
    st.subheader("Metrics Comparison")

    fig = go.Figure()
    for _, row in metrics_df.iterrows():
        fig.add_trace(go.Bar(name=row["Model"], x=["MAE","RMSE"], y=[row["MAE"], row["RMSE"]]))
    fig.update_layout(
        barmode="group", title="MAE & RMSE by Model",
        template="plotly_white", yaxis_title="Error",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full Metrics Table")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Feature Importance (Random Forest)")
    feat_imp = metrics.get("feature_importance", {})
    if feat_imp:
        fi_df = (
            pd.DataFrame(list(feat_imp.items()), columns=["Feature","Importance"])
            .sort_values("Importance", ascending=True)
        )
        fig2 = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            title="Feature Importance",
            color="Importance", color_continuous_scale="Blues",
            template="plotly_white",
        )
        fig2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    ca, cb = st.columns(2)
    ca.metric("Training samples", metrics.get("train_size", "—"))
    cb.metric("Test samples",     metrics.get("test_size",  "—"))
    st.caption(
        f"XGBoost available: **{metrics.get('xgboost_available', False)}** &nbsp;|&nbsp; "
        "80/20 chronological train-test split"
    )
