# E-Commerce Sales Prediction

A full-stack e-commerce analytics and sales prediction project built on the
`basket_details.csv` dataset (~15,000 transactions, June 2019).

---

## Project Structure

```
d:\ibmproject\
├── basket_details.csv          ← source dataset
├── backend/
│   ├── app.py                  ← Flask REST API
│   ├── data_processor.py       ← data loading & feature engineering
│   ├── model.py                ← ML training & prediction
│   └── requirements.txt
├── frontend/
│   ├── app.py                  ← Streamlit dashboard
│   └── requirements.txt
├── models/                     ← saved model artefacts (auto-created)
└── README.md
```

---

## Dataset

| Column        | Type     | Description                          |
|---------------|----------|--------------------------------------|
| customer_id   | int      | Unique customer identifier           |
| product_id    | int      | Unique product identifier            |
| basket_date   | date     | Date of transaction                  |
| basket_count  | int      | Number of items in the basket        |

- **15 001 rows** (2019-05-26 → 2019-06-19, 25 calendar days)

---

## Quick Start

### 1 – Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2 – (Optional) Pre-train the model

```bash
cd backend
python model.py
```

The model is also trained automatically on the first API request.

### 3 – Start the Flask backend

```bash
cd backend
python app.py
# → http://localhost:5000
```

### 4 – Install frontend dependencies (new terminal)

```bash
cd frontend
pip install -r requirements.txt
```

### 5 – Start the Streamlit dashboard

```bash
cd frontend
streamlit run app.py
# → http://localhost:8501
```

---

## API Reference

All endpoints return JSON. Base URL: `http://localhost:5000`

| Method | Endpoint                  | Query params       | Description                          |
|--------|---------------------------|--------------------|--------------------------------------|
| GET    | `/api/health`             | —                  | Liveness check                       |
| GET    | `/api/summary`            | —                  | Dataset summary statistics           |
| GET    | `/api/sales-trend`        | —                  | Daily total sales time series        |
| GET    | `/api/predict`            | `days` (1–90)      | Predict next N days of sales         |
| GET    | `/api/top-products`       | `n` (1–50)         | Top N products by basket volume      |
| GET    | `/api/customer-stats`     | —                  | Customer purchase statistics         |
| GET    | `/api/model-metrics`      | —                  | MAE, RMSE, R², feature importance    |

Example:

```bash
curl "http://localhost:5000/api/predict?days=14"
curl "http://localhost:5000/api/top-products?n=5"
```

---

## ML Model

| Component       | Detail                                            |
|-----------------|---------------------------------------------------|
| Algorithm       | Random Forest + XGBoost ensemble (simple average) |
| Target          | Daily total `basket_count`                        |
| Split           | 80 / 20 chronological train-test                  |
| Features        | See table below                                   |

### Features

| Feature          | Description                                      |
|------------------|--------------------------------------------------|
| day_of_week      | 0 = Monday … 6 = Sunday                         |
| month            | Calendar month (1–12)                            |
| week_of_year     | ISO week number                                  |
| day_of_month     | Day of the month (1–31)                          |
| is_weekend       | 1 if Saturday / Sunday, else 0                   |
| lag_1            | Sales yesterday                                  |
| lag_7            | Sales one week ago                               |
| lag_14           | Sales two weeks ago                              |
| rolling_mean_7   | 7-day rolling average of past sales              |
| rolling_mean_14  | 14-day rolling average of past sales             |

### Saved artefacts (`models/`)

| File                | Contents                                     |
|---------------------|----------------------------------------------|
| `rf_model.joblib`   | Trained RandomForestRegressor                |
| `xgb_model.joblib`  | Trained XGBRegressor                         |
| `model_metrics.json`| MAE, RMSE, R² for RF / XGB / ensemble       |
| `feature_names.json`| Ordered feature column list                  |

---

## Dashboard Pages

| Page                | What it shows                                           |
|---------------------|---------------------------------------------------------|
| 📊 Dashboard        | KPI cards, sales trend line, day-of-week bar chart      |
| 🔮 Sales Prediction | Slider (7–30 days), combined historical + forecast chart |
| 📦 Product Analysis | Top-N products bar chart, customer & transaction counts |
| 👤 Customer Analysis| Top customers, purchase frequency distribution          |
| 🤖 Model Performance| MAE / RMSE / R² comparison, feature importance chart    |

---

## Tech Stack

**Backend**: Python · Flask · pandas · scikit-learn · XGBoost · NumPy · joblib  
**Frontend**: Python · Streamlit · Plotly · requests · pandas
