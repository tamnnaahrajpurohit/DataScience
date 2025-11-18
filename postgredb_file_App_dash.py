# app_postgres_olist_v3.py
import traceback, sys
import os
import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import text
import plotly.express as px
import plotly.graph_objects as go
import requests
from urllib.parse import quote_plus

st.set_page_config(page_title="Company Dashboard (Olist) — Postgres v3", layout="wide")

# ---------- DB config ----------
# Default (fallback) - keep as a safe local default or change to your local DB
DATABASE_URL = "postgresql://postgres:1234@db.hzyzqmyabfqagcxdwjti.supabase.co:5432/postgres"

# ---------- Engine creation with IPv4 fallback and diagnostics ----------
import socket
from urllib.parse import urlparse, urlunparse
# Prefer Streamlit secrets -> environment variables
def _get_supabase_creds():
    supabase_url = None
    supabase_key = None
    try:
        if hasattr(st, "secrets") and st.secrets:
            supabase_url = st.secrets.get("SUPABASE_URL") or st.secrets.get("supabase_url")
            supabase_key = st.secrets.get("SUPABASE_KEY") or st.secrets.get("supabase_key")
    except Exception:
        supabase_url = None
        supabase_key = None
    supabase_url = supabase_url or os.getenv("SUPABASE_URL")
    supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
    return supabase_url, supabase_key

# If Supabase REST creds exist, we'll prefer REST (HTTPS) — works on Streamlit Cloud.
SUPABASE_URL, SUPABASE_KEY = _get_supabase_creds()
# Optional direct DATABASE_URL for local development or hosts that allow outbound Postgres
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:1234@localhost:5432/retail_dashboard")

# Diagnostic: show which approach we'll try (no secrets printed)
if SUPABASE_URL and SUPABASE_KEY:
    st.info("Using Supabase REST (HTTPS) to fetch tables (preferred for hosted apps).")
else:
    st.info("Supabase REST credentials not found. Will attempt direct PostgreSQL via DATABASE_URL (may fail on hosted runtimes).")

# --- REST reader for Supabase PostgREST ---
def read_table_rest(table_name, limit=None):
    """Return DataFrame from Supabase REST endpoint, or empty DataFrame on failure."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return pd.DataFrame()

    endpoint = SUPABASE_URL.rstrip("/") + f"/rest/v1/{quote_plus(table_name)}?select=*"
    if limit and isinstance(limit, int):
        endpoint += f"&limit={limit}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Accept": "application/json",
    }
    try:
        resp = requests.get(endpoint, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            if len(data) == 0:
                return pd.DataFrame()
            return pd.DataFrame(data)
        else:
            # unexpected structure
            st.warning(f"Supabase REST returned unexpected structure for '{table_name}'.")
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"Supabase REST request failed for '{table_name}': {e}")
        return pd.DataFrame()

# --- SQLAlchemy engine creation (only used if REST creds missing or REST fallback fails) ---
engine = None
if not SUPABASE_URL or not SUPABASE_KEY:
    # Only try direct DB when Supabase REST not available (reduces failures on hosted runtimes)
    try:
        from sqlalchemy import create_engine
        engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)
        # quick test
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            st.success("Direct database connection successful (using DATABASE_URL).")
        except Exception as e:
            st.warning("Direct DATABASE_URL exists but connection test failed.")
            st.text(str(e))
            engine = None
    except Exception as e:
        st.error("Failed to create SQLAlchemy engine.")
        st.text(str(e))
        engine = None

# --- Unified read_table that prefers REST, then falls back to SQLAlchemy if available ---
def read_table(table_name, schema="public", limit=None):
    """
    Preferred order:
    1) Supabase REST (HTTPS) if credentials present
    2) SQLAlchemy engine if created
    3) Return empty DataFrame
    """
    # 1) REST
    if SUPABASE_URL and SUPABASE_KEY:
        df = read_table_rest(table_name, limit=limit)
        if not df.empty:
            return df
        # if REST returns empty, we still try SQLAlchemy fallback below (useful for private tables)
    # 2) SQLAlchemy fallback
    if engine is not None:
        try:
            q = text(f'SELECT * FROM "{schema}"."{table_name}"')
            return pd.read_sql(q, engine)
        except Exception:
            try:
                return pd.read_sql(f"select * from {table_name}", engine)
            except Exception:
                st.warning(f"SQLAlchemy fallback failed for '{table_name}'.")
                return pd.DataFrame()
    # 3) nothing available
    st.info(f"No data loaded for '{table_name}' (no REST creds and no DB engine).")
    return pd.DataFrame()

# ---------- helpers ----------
@st.cache_data(ttl=600)

# Place near top of your file (imports)

# New helper: read via Supabase REST (PostgREST)
def read_table_rest(table_name):
    """
    Reads an entire table via Supabase REST (PostgREST) and returns a pandas.DataFrame.
    Requires SUPABASE_URL and SUPABASE_KEY present in st.secrets or env.
    """
    supabase_url = None
    supabase_key = None

    # Prefer Streamlit secrets
    try:
        if hasattr(st, "secrets") and st.secrets is not None:
            supabase_url = st.secrets.get("SUPABASE_URL") or st.secrets.get("supabase_url")
            supabase_key = st.secrets.get("SUPABASE_KEY") or st.secrets.get("supabase_key")
    except Exception:
        supabase_url = None
        supabase_key = None

    # Fallback to env
    supabase_url = supabase_url or os.getenv("SUPABASE_URL")
    supabase_key = supabase_key or os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        # no supabase credentials available
        return pd.DataFrame()

    # Build REST endpoint: /rest/v1/<table>?select=*
    # Use urljoin-like safe concatenation (supabase_url doesn't end with slash normally).
    endpoint = supabase_url.rstrip("/") + f"/rest/v1/{quote_plus(table_name)}?select=*"
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Accept": "application/json",
    }

    try:
        resp = requests.get(endpoint, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # If result is a list of objects, convert to DataFrame
        if isinstance(data, list):
            return pd.DataFrame(data)
        else:
            # Unexpected structure, return empty
            return pd.DataFrame()
    except Exception as e:
        # fallback: if you still have an engine, try SQLAlchemy read (keeps older behavior)
        try:
            if engine is not None:
                q = text(f'SELECT * FROM public."{table_name}"')
                return pd.read_sql(q, engine)
        except Exception:
            pass
        # Log minimal info to UI (don't leak secrets)
        st.write(f"Failed to fetch table '{table_name}' via Supabase REST: {e}")
        return pd.DataFrame()





def safe_to_datetime(s, **kwargs):
    try:
        return pd.to_datetime(s, errors="coerce", **kwargs)
    except Exception:
        return pd.to_datetime(s.astype(str), errors="coerce", **kwargs)

def money(x):
    try:
        return f"R${x:,.0f}"
    except Exception:
        return x

def safe_plot(df, fig, key=None):
    if df is None or (isinstance(df, (pd.DataFrame, pd.Series)) and df.empty) or fig is None:
        st.info("No data to show this chart.")
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)

def make_table_figure(df, max_rows=50):
    if df is None or df.empty:
        return None
    df2 = df.head(max_rows).copy()
    header = list(df2.columns)
    cells = [df2[col].astype(str).tolist() for col in df2.columns]
    fig = go.Figure(data=[go.Table(header=dict(values=header, align="left"),
                                   cells=dict(values=cells, align="left"))])
    fig.update_layout(height=min(500, 60 + 22 * len(df2)))
    return fig

# ---------- load data ----------
@st.cache_data(ttl=600)
def load_all():
    users = read_table("users")
    warehouses = read_table("warehouses")
    products = read_table("products")
    orders = read_table("orders")
    order_items = read_table("order_items")
    reviews = read_table("reviews")
    mkt = read_table("marketing_spend")
    return users, warehouses, products, orders, order_items, reviews, mkt

users, warehouses, products, orders, order_items, reviews, mkt = load_all()

# ---------- Build fact table (merge warehouses to get names) ----------
@st.cache_data(ttl=600)
def build_fact_table(orders, order_items, products, warehouses):
    df = order_items.copy() if isinstance(order_items, pd.DataFrame) else pd.DataFrame()
    if df.empty:
        return df

    df['qty'] = df.get('qty', pd.Series(1, index=df.index)).fillna(1).astype(float)
    df['unit_price'] = pd.to_numeric(df.get('unit_price', pd.Series(dtype=float)), errors='coerce')

    ords = orders.copy() if isinstance(orders, pd.DataFrame) else pd.DataFrame()
    ords['order_date'] = safe_to_datetime(ords.get('order_date', ords.get('order_purchase_timestamp', pd.Series(dtype='datetime64[ns]'))))
    ords['shipping_cost'] = pd.to_numeric(ords.get('shipping_cost', 0), errors='coerce').fillna(0)
    ords['discount_amount'] = pd.to_numeric(ords.get('discount_amount', 0), errors='coerce').fillna(0)
    ords['channel'] = ords.get('channel', 'online')
    ords['warehouse_id'] = ords.get('warehouse_id', pd.NA)

    orders_cols = [c for c in ['order_id','order_date','discount_amount','shipping_cost','channel','warehouse_id','status'] if c in ords.columns]
    df = df.merge(ords[orders_cols], on='order_id', how='left')

    prod = products.copy() if isinstance(products, pd.DataFrame) else pd.DataFrame()
    if prod.empty:
        prod = df[['product_id']].drop_duplicates().assign(unit_price=np.nan, unit_cost=np.nan, sku=df.get('product_id', pd.Series()).astype(str).str[:8], category='unknown')

    prod['unit_price'] = pd.to_numeric(prod.get('unit_price', pd.Series(dtype=float)), errors='coerce')
    prod['unit_cost'] = pd.to_numeric(prod.get('unit_cost', prod.get('unit_price', 0) * 0.7), errors='coerce').fillna(prod.get('unit_price', 0) * 0.7)
    prod['sku'] = prod.get('sku', prod.get('product_id', pd.Series()).astype(str).str[:8])
    # --- after loading/creating prod dataframe and before merging into df ---
    # Normalize / clean category values to avoid inconsistent names
    if 'category' in prod.columns:
        prod['category'] = prod['category'].astype(str).str.strip().replace({'nan':'unknown', 'None':'unknown'})
        prod.loc[prod['category'].str.len() == 0, 'category'] = 'unknown'
    else:
        prod['category'] = 'unknown'

    prod_keep = [c for c in ['product_id','sku','category','unit_cost','unit_price'] if c in prod.columns]
    df = df.merge(prod[prod_keep].rename(columns={'unit_price':'unit_price_prod'}), on='product_id', how='left')

    # after all computations, ensure fact.category exists and is cleaned
    if 'category' in df.columns:
        df['category'] = df['category'].astype(str).str.strip().replace({'nan':'unknown', 'None':'unknown'})
        df.loc[df['category'].str.len() == 0, 'category'] = 'unknown'
    else:
        df['category'] = 'unknown'

    prod_keep = [c for c in ['product_id','sku','category','unit_cost','unit_price'] if c in prod.columns]
    df = df.merge(prod[prod_keep].rename(columns={'unit_price':'unit_price_prod'}), on='product_id', how='left')

    df['unit_price'] = df['unit_price'].fillna(df.get('unit_price_prod', np.nan))
    df['line_total'] = df['qty'] * df['unit_price']

    line_sum = df.groupby('order_id')['line_total'].transform('sum').replace(0, np.nan)
    df['discount_share'] = (df['line_total'] / line_sum) * df['discount_amount'].fillna(0)
    df['net_line_revenue'] = df['line_total'] - df['discount_share'].fillna(0)
    ship_sum = df.groupby('order_id')['line_total'].transform('sum').replace(0, np.nan)
    df['ship_share'] = (df['line_total'] / ship_sum) * df['shipping_cost'].fillna(0)

    # --- FIX: compute cogs safely even if 'unit_cost' column missing ---
    if 'unit_cost' in df.columns:
        unit_cost_series = df['unit_cost'].fillna(df.get('unit_price', 0) * 0.7)
    else:
        unit_cost_series = df.get('unit_price', pd.Series(dtype=float)).fillna(0) * 0.7
    df['cogs'] = (df['qty'] * unit_cost_series).fillna(0)
    # --- end FIX ---

    df['gross_profit'] = df['net_line_revenue'] - df['cogs'] - df['ship_share']
    df['margin_pct'] = np.where(df['net_line_revenue']!=0, df['gross_profit']/df['net_line_revenue'], 0)

    wh = warehouses.copy() if isinstance(warehouses, pd.DataFrame) else pd.DataFrame()
    if not wh.empty:
        keep = [c for c in ['warehouse_id','name','city','state'] if c in wh.columns]
        wh_subset = wh[keep].rename(columns={'name':'warehouse_name','city':'warehouse_city','state':'warehouse_state'})
        df = df.merge(wh_subset, on='warehouse_id', how='left')
    else:
        df['warehouse_name'] = df['warehouse_id']
        df['warehouse_city'] = np.nan
        df['warehouse_state'] = np.nan

    if 'order_date' not in df.columns:
        df['order_date'] = ords.get('order_date', pd.NaT)

    return df

fact = build_fact_table(orders, order_items, products, warehouses)

# ---------- Sidebar filters (extended) ----------
st.sidebar.header("Filters — Advanced")
if isinstance(orders, pd.DataFrame) and 'order_date' in orders.columns and not orders['order_date'].isna().all():
    min_date = safe_to_datetime(orders['order_date']).min()
    max_date = safe_to_datetime(orders['order_date']).max()
else:
    min_date, max_date = pd.to_datetime("2000-01-01"), pd.to_datetime("2100-01-01")
date_range = st.sidebar.date_input("Order Date Range", value=(min_date, max_date))

channels = ["All"] + sorted(orders['channel'].dropna().unique().tolist()) if isinstance(orders, pd.DataFrame) and 'channel' in orders.columns else ["All"]
channel_sel = st.sidebar.selectbox("Channel", channels, index=0)

# ---------- Sidebar: Warehouse name list (safe) ----------
if isinstance(warehouses, pd.DataFrame) and 'name' in warehouses.columns:
    # prefer human-friendly warehouse names when available
    wh_names = ["All"] + sorted(warehouses['name'].dropna().astype(str).unique().tolist())
else:
    # only add warehouse ids from orders if the column exists
    if isinstance(orders, pd.DataFrame) and 'warehouse_id' in orders.columns:
        wh_names = ["All"] + sorted(orders['warehouse_id'].dropna().astype(str).unique().tolist())
    else:
        # neither warehouses.name nor orders.warehouse_id exists — use just "All"
        wh_names = ["All"]
warehouse_name_sel = st.sidebar.selectbox("Warehouse (Seller name)", wh_names, index=0)


wh_cities = ["All"] + sorted(warehouses['city'].dropna().astype(str).unique().tolist()) if isinstance(warehouses, pd.DataFrame) and 'city' in warehouses.columns else ["All"]
warehouse_city_sel = st.sidebar.selectbox("Warehouse City", wh_cities, index=0)

wh_states = ["All"] + sorted(warehouses['state'].dropna().astype(str).unique().tolist()) if isinstance(warehouses, pd.DataFrame) and 'state' in warehouses.columns else ["All"]
warehouse_state_sel = st.sidebar.selectbox("Warehouse State", wh_states, index=0)

cats = ["All"] + sorted(products['category'].dropna().unique().tolist()) if isinstance(products, pd.DataFrame) and 'category' in products.columns else ["All"]
cat_sel = st.sidebar.selectbox("Product Category", cats, index=0)

sku_text = st.sidebar.text_input("SKU contains (text, leave empty = all)")
product_id_text = st.sidebar.text_input("Product ID contains (text)")

rating_min, rating_max = st.sidebar.slider("Rating range", 0, 5, (0,5))
min_rev = st.sidebar.number_input("Min revenue per order (R$)", value=0.0, step=100.0)
max_rev = st.sidebar.number_input("Max revenue per order (R$) (0 = no max)", value=0.0, step=100.0)
qty_min = st.sidebar.number_input("Min qty per line", value=0, step=1)
qty_max = st.sidebar.number_input("Max qty per line (0 = no max)", value=0, step=1)

# ---------- Apply filters ----------
f = fact.copy() if isinstance(fact, pd.DataFrame) else pd.DataFrame()
if not f.empty:
    f['order_date'] = safe_to_datetime(f['order_date'])
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        s, e = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        f = f[(f['order_date'] >= s) & (f['order_date'] <= e)]
    if channel_sel != "All" and 'channel' in f.columns:
        f = f[f['channel'] == channel_sel]
    if warehouse_name_sel != "All" and 'warehouse_name' in f.columns:
        f = f[f['warehouse_name'].astype(str) == str(warehouse_name_sel)]
    if warehouse_city_sel != "All" and 'warehouse_city' in f.columns:
        f = f[f['warehouse_city'].astype(str) == str(warehouse_city_sel)]
    if warehouse_state_sel != "All" and 'warehouse_state' in f.columns:
        f = f[f['warehouse_state'].astype(str) == str(warehouse_state_sel)]
    if cat_sel != "All" and 'category' in f.columns:
        f = f[f['category'] == cat_sel]
    if sku_text:
        f = f[f['sku'].astype(str).str.contains(sku_text, case=False, na=False)]
    if product_id_text:
        f = f[f['product_id'].astype(str).str.contains(product_id_text, case=False, na=False)]
    if qty_min > 0:
        f = f[f['qty'] >= qty_min]
    if qty_max > 0:
        f = f[f['qty'] <= qty_max]
    if min_rev > 0 or max_rev > 0:
        ord_rev = f.groupby("order_id")["net_line_revenue"].sum().rename("rev_per_order").reset_index()
        if min_rev > 0:
            keep_ids = ord_rev[ord_rev['rev_per_order'] >= min_rev]['order_id']
            f = f[f['order_id'].isin(keep_ids)]
        if max_rev > 0:
            keep_ids = ord_rev[ord_rev['rev_per_order'] <= max_rev]['order_id']
            f = f[f['order_id'].isin(keep_ids)]

# ---------- Dashboard header ----------
st.title("Company Dashboard — Insights & Profitability (Postgres v3)")

if f.empty:
    st.warning("No data in fact after applying filters — check tables and filter selections.")
else:
    total_rev = f["net_line_revenue"].sum()
    gross_profit = f["gross_profit"].sum()
    orders_count = f["order_id"].nunique()
    aov = f.groupby("order_id")["net_line_revenue"].sum().mean()
    margin_pct = (gross_profit / total_rev) if total_rev != 0 else 0

    rr = 0.0
    if isinstance(orders, pd.DataFrame) and "status" in orders.columns:
        o2 = orders.copy()
        o2['is_returned'] = o2['status'].isin(["canceled", "unavailable"]).astype(int)
        tot = o2['order_id'].nunique()
        ret = o2[o2['is_returned'] == 1]['order_id'].nunique()
        rr = (ret / tot) if tot > 0 else 0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Revenue (filtered)", money(total_rev))
    k2.metric("Gross Profit", money(gross_profit))
    k3.metric("Gross Margin %", f"{margin_pct*100:.2f}%")
    k4.metric("Orders", f"{orders_count:,}")
    k5.metric("AOV", money(aov) if not np.isnan(aov) else "N/A")
    k6.metric("Return Rate", f"{rr*100:.2f}%")

st.markdown("---")

# ---------- Chart generator helpers ----------
def chart_weekly_revenue(df):
    if 'order_date' in df.columns and 'net_line_revenue' in df.columns:
        ts = df.groupby(pd.Grouper(key='order_date', freq='W'))['net_line_revenue'].sum().reset_index()
        return px.line(ts, x='order_date', y='net_line_revenue', markers=True, title='Revenue (Weekly)')
    return None

def chart_cumulative_revenue(df):
    if 'order_date' in df.columns and 'net_line_revenue' in df.columns:
        ts = df.groupby(pd.Grouper(key='order_date', freq='D'))['net_line_revenue'].sum().cumsum().reset_index(name='cum_rev')
        return px.line(ts, x='order_date', y='cum_rev', title='Cumulative Revenue (Daily)')
    return None

def chart_monthly_rev_vs_gp(df):
    if 'month' not in df.columns:
        if 'order_date' in df.columns:
            df['month'] = df['order_date'].dt.to_period('M').astype(str)
        else:
            return None
    if 'net_line_revenue' in df.columns and 'gross_profit' in df.columns:
        rev_m = df.groupby('month')['net_line_revenue'].sum().reset_index(name='revenue')
        gp_m = df.groupby('month')['gross_profit'].sum().reset_index(name='gross_profit')
        both = rev_m.merge(gp_m, on='month', how='left').fillna(0)
        return px.bar(both, x='month', y=['revenue','gross_profit'], barmode='group', title='Revenue vs Gross Profit (Monthly)')
    return None

def chart_rev_distribution(df):
    if 'net_line_revenue' in df.columns:
        per_order = df.groupby('order_id')['net_line_revenue'].sum().reset_index()
        return px.histogram(per_order, x='net_line_revenue', nbins=60, title='Revenue per Order Distribution')
    return None

def chart_aov_monthly(df):
    if 'month' not in df.columns:
        if 'order_date' in df.columns:
            df['month'] = df['order_date'].dt.to_period('M').astype(str)
        else:
            return None
    if 'net_line_revenue' in df.columns:
        aov_by_month = df.groupby('month').apply(lambda x: x.groupby('order_id')['net_line_revenue'].sum().mean()).reset_index(name='AOV')
        return px.line(aov_by_month, x='month', y='AOV', markers=True, title='AOV (Monthly)')
    return None

def chart_top_products(df, topn=15):
    if all(c in df.columns for c in ['product_id','sku','net_line_revenue']):
        prod_sum = df.groupby(['product_id','sku'])['net_line_revenue'].sum().reset_index().sort_values('net_line_revenue', ascending=False).head(topn)
        return px.bar(prod_sum, x='sku', y='net_line_revenue', title=f'Top {topn} Products by Revenue')
    return None

def chart_top_categories(df, topn=15):
    if df is None or df.empty:
        if isinstance(products, pd.DataFrame) and 'category' in products.columns:
            prod = products.copy()
            prod['category'] = prod['category'].astype(str).str.strip().replace({'nan':'unknown','None':'unknown'})
            prod.loc[prod['category'].str.len() == 0, 'category'] = 'unknown'
            cat = prod['category'].value_counts().head(topn).reset_index()
            cat.columns = ['category','count']
            return px.bar(cat, x='category', y='count', title=f'Top {topn} Categories (products fallback)')
        return None

    if 'category' not in df.columns or df['category'].isna().all():
        if isinstance(products, pd.DataFrame) and 'category' in products.columns:
            p = products.copy()
            p['category'] = p['category'].astype(str).str.strip().replace({'nan':'unknown','None':'unknown'})
            p.loc[p['category'].str.len() == 0, 'category'] = 'unknown'
            cat = p['category'].value_counts().head(topn).reset_index()
            cat.columns = ['category','count']
            return px.bar(cat, x='category', y='count', title=f'Top {topn} Categories (products fallback)')
        return None

    df2 = df.copy()
    df2['category'] = df2['category'].astype(str).str.strip().replace({'nan':'unknown','None':'unknown'})
    df2.loc[df2['category'].str.len() == 0, 'category'] = 'unknown'

    cat = df2.groupby('category')['net_line_revenue'].sum().reset_index().sort_values('net_line_revenue', ascending=False).head(topn)
    if cat['net_line_revenue'].abs().sum() == 0:
        cat = df2['category'].value_counts().head(topn).reset_index()
        cat.columns = ['category','count']
        return px.bar(cat, x='category', y='count', title=f'Top {topn} Categories (by count)')
    else:
        return px.bar(cat, x='category', y='net_line_revenue', title=f'Top {topn} Categories by Revenue')


def chart_margin_hist(df):
    if 'margin_pct' in df.columns:
        return px.histogram(df, x='margin_pct', nbins=50, title='Margin % Distribution')
    return None

def chart_qty_distribution(df):
    if 'qty' in df.columns:
        return px.histogram(df, x='qty', nbins=40, title='Quantity per Line Distribution')
    return None

def chart_rev_vs_shipping(df):
    if 'net_line_revenue' in df.columns and 'ship_share' in df.columns:
        ord_rev = df.groupby('order_id')[['net_line_revenue','ship_share']].sum().reset_index()
        return px.scatter(ord_rev, x='net_line_revenue', y='ship_share', title='Shipping vs Order Revenue')
    return None

def chart_discount_vs_rev(df):
    if 'net_line_revenue' in df.columns and 'discount_share' in df.columns:
        ord_rev = df.groupby('order_id')[['net_line_revenue','discount_share']].sum().reset_index()
        return px.scatter(ord_rev, x='net_line_revenue', y='discount_share', title='Discount vs Order Revenue')
    return None

def chart_rev_by_warehouse(df):
    if 'warehouse_name' in df.columns and 'net_line_revenue' in df.columns:
        wh = df.groupby('warehouse_name')['net_line_revenue'].sum().reset_index().sort_values('net_line_revenue', ascending=False).head(20)
        return px.bar(wh, x='warehouse_name', y='net_line_revenue', title='Revenue by Warehouse (Top 20)')
    return None

def chart_gp_by_warehouse(df):
    if 'warehouse_name' in df.columns and 'gross_profit' in df.columns:
        wh = df.groupby('warehouse_name')['gross_profit'].sum().reset_index().sort_values('gross_profit', ascending=False).head(20)
        return px.bar(wh, x='warehouse_name', y='gross_profit', title='Gross Profit by Warehouse (Top 20)')
    return None

def chart_revenue_pareto(df):
    if 'sku' in df.columns and 'net_line_revenue' in df.columns:
        sku_rev = df.groupby('sku')['net_line_revenue'].sum().reset_index().sort_values('net_line_revenue', ascending=False)
        sku_rev['cum_pct'] = sku_rev['net_line_revenue'].cumsum() / sku_rev['net_line_revenue'].sum()
        return px.line(sku_rev.reset_index().rename(columns={'index':'rank'}), x='rank', y='cum_pct', title='Cumulative revenue Pareto (by SKU)')
    return None

def chart_rating_vs_return(df, orders_df, reviews_df):
    if reviews_df is None or reviews_df.empty or orders_df is None or orders_df.empty:
        return None
    if 'rating' in reviews_df.columns and 'order_id' in reviews_df.columns and 'status' in orders_df.columns:
        o = orders_df[['order_id','status']].copy()
        o['is_returned'] = o['status'].isin(['canceled','unavailable']).astype(int)
        rj = reviews_df.merge(o, on='order_id', how='left')
        byrat = rj.groupby('rating')['is_returned'].mean().reset_index()
        return px.line(byrat, x='rating', y='is_returned', markers=True, title='Return Rate vs Rating')
    return None

def chart_reviews_per_month(reviews_df):
    if reviews_df is None or reviews_df.empty:
        return None
    if 'review_date' in reviews_df.columns:
        r = reviews_df.copy()
        r['rm'] = safe_to_datetime(r['review_date']).dt.to_period('M').astype(str)
        rpm = r.groupby('rm')['review_id'].nunique().reset_index(name='reviews')
        return px.line(rpm, x='rm', y='reviews', markers=True, title='Reviews per Month')
    return None

def chart_top_low_rated_skus(reviews_df):
    if reviews_df is None or reviews_df.empty:
        return None
    if 'rating' in reviews_df.columns and 'product_id' in reviews_df.columns:
        if 'sku' in reviews_df.columns:
            low = reviews_df[reviews_df['rating']<=2].groupby('sku')['rating'].count().reset_index(name='low_count').sort_values('low_count', ascending=False).head(20)
            return px.bar(low, x='sku', y='low_count', title='Top low-rated SKUs')
        else:
            return None
    return None

def chart_marketing_spend(mkt_df):
    if mkt_df is None or mkt_df.empty:
        return None
    try:
        mkt = mkt_df.copy()
        mkt['month'] = safe_to_datetime(mkt['month']).dt.to_period('M').astype(str)
        return px.bar(mkt, x='month', y='spend', color='channel', barmode='group', title='Marketing Spend by Channel (Monthly)')
    except Exception:
        return None

def chart_cpa_by_channel(mkt_df, fact_df):
    if mkt_df is None or mkt_df.empty or fact_df is None or fact_df.empty:
        return None
    try:
        mkt = mkt_df.copy()
        mkt['month'] = safe_to_datetime(mkt['month']).dt.to_period('M').astype(str)
        fact = fact_df.copy()
        fact['month'] = fact['order_date'].dt.to_period('M').astype(str)
        ord_by_chan = fact.groupby(['month','channel'])['order_id'].nunique().reset_index(name='orders')
        mk = mkt.merge(ord_by_chan, on=['month','channel'], how='left').fillna({'orders':0})
        mk['cpa'] = mk['spend'] / mk['orders'].replace(0, np.nan)
        return px.bar(mk, x='month', y='cpa', color='channel', barmode='group', title='CPA by Channel')
    except Exception:
        return None

def chart_top_customers(f_df, users_df, topn=20):
    if f_df is None or f_df.empty:
        return None
    if 'user_id' in f_df.columns:
        cust = f_df.groupby('user_id').agg(revenue=('net_line_revenue','sum'), orders=('order_id','nunique')).reset_index().sort_values('revenue', ascending=False).head(topn)
        if users_df is not None and 'user_id' in users_df.columns:
            cust = cust.merge(users_df[['user_id','city']], on='user_id', how='left')
        return px.bar(cust, x='user_id', y='revenue', hover_data=['orders','city'], title='Top customers by revenue (Top {})'.format(topn))
    return None

def chart_orders_per_month(df):
    if 'order_date' in df.columns:
        df['month'] = df['order_date'].dt.to_period('M').astype(str)
        ord_m = df.groupby('month')['order_id'].nunique().reset_index(name='orders')
        return px.line(ord_m, x='month', y='orders', markers=True, title='Orders per Month')
    return None

def chart_rev_heatmap_warehouse_month(df):
    if 'warehouse_name' in df.columns and 'month' in df.columns and 'net_line_revenue' in df.columns:
        whm = df.groupby(['month','warehouse_name'])['net_line_revenue'].sum().reset_index()
        try:
            fig = px.density_heatmap(whm, x='month', y='warehouse_name', z='net_line_revenue', labels={'net_line_revenue':'revenue'}, title='Revenue heatmap (warehouse x month)')
            return fig
        except Exception:
            return None
    return None

def chart_orders_by_status(orders_df):
    if orders_df is None or orders_df.empty or 'status' not in orders_df.columns:
        return None
    s = orders_df['status'].value_counts().reset_index()
    s.columns = ['status','count']
    return px.bar(s, x='status', y='count', title='Orders by Status')
    
def chart_repeat_purchase_rate(orders_df):
    if orders_df is None or orders_df.empty or 'user_id' not in orders_df.columns:
        return None
    ob = orders_df.groupby('user_id')['order_id'].nunique().reset_index(name='orders_per_user')
    repeat_rate = (ob['orders_per_user']>1).mean()
    kdf = pd.DataFrame({'metric':['repeat_rate'], 'value':[repeat_rate]})
    return px.bar(kdf, x='metric', y='value', title='Repeat Purchase Rate')

# ---------- Render function: runs up to max_charts generators and prints them ----------
def render_charts_for_tab(generators, tab_name="tab", max_charts=25):
    """
    generators: list of callables that return a plotly figure (or None)
    tab_name: string used to create unique element keys
    """
    shown = 0
    gen_idx = 0

    for gen in generators:
        if shown >= max_charts:
            break
        gen_idx += 1
        try:
            fig = gen()
        except Exception:
            fig = None

        # If generator returned a callable by mistake, try calling once more
        if callable(fig) and not isinstance(fig, (go.Figure, dict, list, tuple)):
            try:
                fig = fig()
            except Exception:
                fig = None

        # Validate fig is Plotly-compatible
        if fig is None:
            continue
        if not isinstance(fig, (go.Figure, dict, list, tuple)):
            # not a valid figure-like object — skip safely
            continue

        key = f"plot_{tab_name}_{gen_idx}"
        try:
            st.plotly_chart(fig, use_container_width=True, key=key)
        except Exception:
            # If even now it fails (extremely rare), skip the plot to avoid breaking the app
            # Optionally: log the failure to Streamlit so you can debug
            st.write(f"Skipped a chart (invalid figure) for key={key}")
            continue

        shown += 1

    if shown == 0:
        st.info("No charts were generated for this tab (check data availability).")
    elif shown < max_charts:
        st.info(f"Generated {shown} charts (data limited). Requested {max_charts} charts per tab.")

# ---------- Tabs ----------
tab_overview, tab_products, tab_customers, tab_ops, tab_marketing, tab_reviews, tab_warehouses = st.tabs(
    ["Overview","Products & Categories","Customers","Operations","Marketing","Reviews","Warehouses"]
)

# Prepare commonly-used closures so generator functions can reference current data
def overview_generators():
    gens = [
        lambda: chart_weekly_revenue(f),
        lambda: chart_cumulative_revenue(f),
        lambda: chart_monthly_rev_vs_gp(f),
        lambda: chart_rev_distribution(f),
        lambda: chart_aov_monthly(f),
        lambda: chart_rev_by_warehouse(f),
        lambda: chart_gp_by_warehouse(f),
        lambda: chart_revenue_pareto(f),
        lambda: chart_qty_distribution(f),
        lambda: chart_margin_hist(f),
        lambda: chart_rev_vs_shipping(f),
        lambda: chart_discount_vs_rev(f),
        lambda: chart_rev_heatmap_warehouse_month(f),
        lambda: chart_orders_per_month(f),
        lambda: chart_top_products(f, topn=10),
        lambda: chart_top_categories(f, topn=10),
        lambda: chart_top_products(f, topn=25),
        lambda: chart_top_categories(f, topn=25),
        lambda: chart_repeat_purchase_rate(orders),
        lambda: chart_top_customers(f, users, topn=15),
        lambda: chart_top_customers(f, users, topn=50),
        lambda: chart_revenue_pareto(f),
        lambda: chart_weekly_revenue(f),
        lambda: chart_cumulative_revenue(f),
        lambda: chart_monthly_rev_vs_gp(f)
    ]
    return gens

def products_generators():
    gens = [
        lambda: chart_top_products(f, topn=15),
        lambda: chart_top_products(f, topn=25),
        lambda: chart_margin_hist(f),
        lambda: chart_qty_distribution(f),
        lambda: chart_revenue_pareto(f),
        lambda: chart_top_categories(f, topn=15),
        lambda: chart_top_categories(f, topn=25),
        lambda: (px.scatter(products, x='unit_price', y='unit_cost', hover_data=['sku','category'], title='Unit Price vs Unit Cost') if isinstance(products, pd.DataFrame) and 'unit_price' in products.columns and 'unit_cost' in products.columns else None),
        lambda: (px.histogram(products, x='unit_price', nbins=50, title='Product Unit Price Distribution') if isinstance(products, pd.DataFrame) and 'unit_price' in products.columns else None),
        lambda: (px.histogram(products, x='unit_cost', nbins=50, title='Product Unit Cost Distribution') if isinstance(products, pd.DataFrame) and 'unit_cost' in products.columns else None),
        lambda: chart_top_products(f, topn=50),
        lambda: (make_table_figure(products[['product_id','sku','category','unit_price','unit_cost']].drop_duplicates().sort_values('sku') if not products.empty else pd.DataFrame())),
        lambda: chart_margin_hist(f),
        lambda: chart_revenue_pareto(f),
        lambda: chart_rev_distribution(f),
        lambda: chart_top_products(f, topn=5),
        lambda: chart_top_products(f, topn=30),
        lambda: chart_top_categories(f, topn=30),
        lambda: (px.box(f, x='sku', y='margin_pct', title='Margin % by SKU') if 'sku' in f.columns and 'margin_pct' in f.columns else None),
        lambda: (px.box(f, x='category', y='margin_pct', title='Margin % by Category') if 'category' in f.columns and 'margin_pct' in f.columns else None),
        lambda: (px.histogram(f, x='line_total', nbins=50, title='Line Total Distribution') if 'line_total' in f.columns else None),
        lambda: (px.scatter(f.groupby('sku').agg(qty=('qty','sum'), revenue=('net_line_revenue','sum')).reset_index(), x='qty', y='revenue', hover_data=['sku'], title='Qty vs Revenue (SKU)') if 'sku' in f.columns else None),
        lambda: (px.scatter(f.groupby('category').agg(qty=('qty','sum'), revenue=('net_line_revenue','sum')).reset_index(), x='qty', y='revenue', hover_data=['category'], title='Qty vs Revenue (Category)') if 'category' in f.columns else None)
    ]
    return gens

def customers_generators():
    gens = [
        lambda: chart_orders_per_month(f),
        lambda: (px.bar(users['city'].value_counts().reset_index().rename(columns={'index':'city','city':'users'}).head(20), x='index', y='city', title='Top Cities by Registered Users') if isinstance(users, pd.DataFrame) and 'city' in users.columns else None),
        lambda: chart_top_customers(f, users, topn=20),
        lambda: chart_top_customers(f, users, topn=50),
        lambda: (px.histogram(f.groupby('order_id')['order_date'].apply(lambda x: (x.max()-x.min()).days if len(x)>1 else 0).reset_index(name='days_between_orders')['days_between_orders'], nbins=40, title='Days between first and last order per order_id') if 'order_date' in f.columns else None),
        lambda: (px.histogram(f.groupby('user_id').agg(orders=('order_id','nunique'))['orders'], nbins=20, title='Orders per user distribution') if 'user_id' in f.columns else None),
        lambda: (px.line(f.groupby(f['order_date'].dt.to_period('M').astype(str))['net_line_revenue'].sum().reset_index(name='rev'), x=0, y='rev', title='Revenue by Month') if 'order_date' in f.columns else None),
        lambda: (make_table_figure(users[['user_id','city','state']].drop_duplicates()) if isinstance(users, pd.DataFrame) and not users.empty else None),
        lambda: chart_repeat_purchase_rate(orders),
        lambda: (px.bar(f.groupby('user_id').agg(revenue=('net_line_revenue','sum')).reset_index().sort_values('revenue', ascending=False).head(20), x='user_id', y='revenue', title='Top 20 Users by Revenue') if 'user_id' in f.columns else None),
        lambda: (px.box(f, x='user_id', y='net_line_revenue', title='Order revenue distribution by user') if 'user_id' in f.columns else None),
        lambda: (px.scatter(f.groupby('user_id').agg(revenue=('net_line_revenue','sum'), orders=('order_id','nunique')).reset_index(), x='orders', y='revenue', hover_data=['user_id'], title='Orders vs Revenue (user)') if 'user_id' in f.columns else None),
        lambda: chart_rev_distribution(f),
        lambda: chart_aov_monthly(f),
        lambda: chart_top_products(f, topn=10),
        lambda: chart_top_categories(f, topn=10),
        lambda: (px.bar(users['state'].value_counts().reset_index().head(20).rename(columns={'index':'state','state':'users'}), x='index', y='state', title='Top States by Registered Users') if isinstance(users, pd.DataFrame) and 'state' in users.columns else None),
        lambda: (px.histogram(f[f['net_line_revenue']>0]['net_line_revenue'], nbins=50, title='Positive Revenue per Order Distribution') if 'net_line_revenue' in f.columns else None),
        lambda: (px.bar(f.groupby(f['order_date'].dt.to_period('M').astype(str))['order_id'].nunique().reset_index(name='orders'), x='month', y='orders', title='Orders per Month') if 'order_date' in f.columns else None),
        lambda: chart_top_customers(f, users, topn=5),
        lambda: chart_top_customers(f, users, topn=10),
        lambda: chart_top_products(f, topn=5),
        lambda: chart_top_products(f, topn=20)
    ]
    return gens

def ops_generators():
    gens = [
        lambda: chart_orders_by_status(orders),
        lambda: chart_weekly_revenue(f),
        lambda: chart_orders_per_month(f),
        lambda: chart_rev_vs_shipping(f),
        lambda: chart_discount_vs_rev(f),
        lambda: chart_rev_distribution(f),
        lambda: chart_qty_distribution(f),
        lambda: (px.histogram(f, x='ship_share', nbins=50, title='Shipping share distribution') if 'ship_share' in f.columns else None),
        lambda: (px.histogram(f, x='discount_share', nbins=50, title='Discount share distribution') if 'discount_share' in f.columns else None),
        lambda: chart_rev_heatmap_warehouse_month(f),
        lambda: chart_gp_by_warehouse(f),
        lambda: chart_rev_by_warehouse(f),
        lambda: (px.box(f, x='warehouse_name', y='ship_share', title='Shipping share by warehouse') if 'warehouse_name' in f.columns and 'ship_share' in f.columns else None),
        lambda: (px.bar(f.groupby('warehouse_name').agg(returns=('order_id','nunique')).reset_index().head(20), x='warehouse_name', y='returns', title='Orders by warehouse') if 'warehouse_name' in f.columns else None),
        lambda: chart_revenue_pareto(f),
        lambda: chart_top_products(f, topn=10),
        lambda: (px.line(f.groupby(pd.Grouper(key='order_date', freq='D'))['net_line_revenue'].sum().rolling(7).mean().reset_index(name='rev7').dropna(), x='order_date', y='rev7', title='7-day rolling revenue') if 'order_date' in f.columns else None),
        lambda: (px.histogram(f, x='cogs', nbins=50, title='COGS distribution') if 'cogs' in f.columns else None),
        lambda: (px.scatter(f, x='unit_price', y='unit_cost', title='Unit Price vs Unit Cost (lines)') if 'unit_price' in f.columns and 'unit_cost' in f.columns else None),
        lambda: chart_margin_hist(f),
        lambda: chart_top_categories(f, topn=10),
        lambda: chart_top_categories(f, topn=20),
        lambda: chart_top_products(f, topn=30),
        lambda: chart_rev_distribution(f)
    ]
    return gens

def marketing_generators():
    gens = [
        lambda: chart_marketing_spend(mkt),
        # fixed placeholder -> return None directly (was returning a callable previously)
        lambda: None,
        lambda: (make_table_figure(mkt) if isinstance(mkt, pd.DataFrame) and not mkt.empty else None),
        lambda: chart_cpa_by_channel(mkt, f),
        lambda: (px.line(f.groupby(f['order_date'].dt.to_period('M').astype(str))['net_line_revenue'].sum().reset_index(name='rev'), x='month', y='rev', title='Monthly revenue (for marketing)') if 'order_date' in f.columns else None),
        lambda: chart_top_products(f, topn=10),
        lambda: chart_top_categories(f, topn=10),
        lambda: chart_rev_by_warehouse(f),
        lambda: chart_rev_distribution(f),
        lambda: (px.scatter(f.groupby('channel').agg(revenue=('net_line_revenue','sum')).reset_index(), x='channel', y='revenue', title='Revenue by Channel') if 'channel' in f.columns else None),
        lambda: (px.bar(mkt.groupby('channel')['spend'].sum().reset_index(), x='channel', y='spend', title='Total Spend by Channel') if isinstance(mkt, pd.DataFrame) and 'channel' in mkt.columns else None),
        lambda: chart_cpa_by_channel(mkt, f),
        lambda: (px.scatter(f, x='net_line_revenue', y='qty', title='Revenue vs Qty') if 'net_line_revenue' in f.columns and 'qty' in f.columns else None),
        lambda: chart_revenue_pareto(f),
        lambda: chart_top_customers(f, users, topn=20),
        lambda: chart_top_products(f, topn=25),
        lambda: chart_top_categories(f, topn=25),
        lambda: chart_rev_distribution(f),
        lambda: (px.histogram(mkt, x='spend', nbins=40, title='Marketing Spend Distribution') if isinstance(mkt, pd.DataFrame) and 'spend' in mkt.columns else None),
        lambda: (px.line(mkt.groupby(pd.to_datetime(mkt['month']).dt.to_period('M').astype(str))['spend'].sum().reset_index(name='spend'), x='month', y='spend', title='Monthly spend trend') if isinstance(mkt, pd.DataFrame) and 'month' in mkt.columns else None),
        lambda: chart_cpa_by_channel(mkt, f),
        lambda: chart_revenue_pareto(f),
        lambda: chart_top_categories(f, topn=15),
        lambda: chart_top_products(f, topn=15),
        lambda: chart_rev_distribution(f)
    ]
    return gens

def reviews_generators():
    gens = [
        lambda: chart_rating_vs_return(f, orders, reviews),
        lambda: chart_reviews_per_month(reviews),
        lambda: chart_top_low_rated_skus(reviews),
        lambda: (px.bar(reviews['rating'].value_counts().sort_index().reset_index().rename(columns={'index':'rating','rating':'count'}), x='index', y='rating', title='Rating Distribution') if isinstance(reviews, pd.DataFrame) and 'rating' in reviews.columns else None),
        lambda: (make_table_figure(reviews[['review_id','order_id','product_id','rating','review_text']].head(100)) if isinstance(reviews, pd.DataFrame) and 'review_text' in reviews.columns else None),
        lambda: chart_top_products(f, topn=10),
        lambda: chart_rev_distribution(f),
        lambda: chart_top_categories(f, topn=10),
        lambda: (px.scatter(reviews.merge(f.groupby('order_id').agg(order_rev=('net_line_revenue','sum')).reset_index(), on='order_id', how='left'), x='rating', y='order_rev', title='Order revenue by rating') if 'order_id' in reviews.columns else None),
        lambda: (px.bar(reviews['product_id'].value_counts().head(20).reset_index().rename(columns={'index':'product_id','product_id':'count'}), x='index', y='product_id', title='Top reviewed products') if isinstance(reviews, pd.DataFrame) else None),
        lambda: chart_reviews_per_month(reviews),
        lambda: (px.histogram(reviews, x='rating', nbins=6, title='Rating histogram') if isinstance(reviews, pd.DataFrame) and 'rating' in reviews.columns else None),
        lambda: (px.scatter(reviews.merge(products[['product_id','sku']], on='product_id', how='left'), x='rating', y='sku', title='Rating vs SKU') if 'product_id' in reviews.columns and not products.empty else None),
        lambda: chart_top_low_rated_skus(reviews),
        lambda: chart_top_products(f, topn=20),
        lambda: chart_revenue_pareto(f),
        lambda: chart_rev_heatmap_warehouse_month(f),
        lambda: chart_top_customers(f, users, topn=10),
        lambda: (px.histogram(reviews['rating'], nbins=6, title='Rating distribution (histogram)') if isinstance(reviews, pd.DataFrame) and 'rating' in reviews.columns else None),
        lambda: (make_table_figure(reviews[['review_id','rating','review_text']].sample(min(50, len(reviews))) if isinstance(reviews, pd.DataFrame) and 'review_text' in reviews.columns else pd.DataFrame())),
        lambda: chart_top_categories(f),
        lambda: chart_margin_hist(f),
        lambda: chart_qty_distribution(f),
        lambda: chart_rev_distribution(f)
    ]
    return gens

def warehouses_generators():
    gens = [
        lambda: chart_rev_by_warehouse(f),
        lambda: chart_gp_by_warehouse(f),
        lambda: (px.scatter(f.groupby('warehouse_name').agg(revenue=('net_line_revenue','sum'), gross_profit=('gross_profit','sum'), orders=('order_id','nunique')).reset_index(), x='revenue', y='gross_profit', size='orders', hover_data=['warehouse_name'], title='Revenue vs Profit (warehouse)') if 'warehouse_name' in f.columns else None),
        lambda: (make_table_figure(f.groupby(['warehouse_name','warehouse_city','warehouse_state']).agg(revenue=('net_line_revenue','sum'), orders=('order_id','nunique')).reset_index().sort_values('revenue', ascending=False).head(100)) if not f.empty and 'warehouse_name' in f.columns else None),
        lambda: chart_rev_heatmap_warehouse_month(f),
        lambda: (px.bar(f.groupby('warehouse_city')['net_line_revenue'].sum().reset_index().sort_values('net_line_revenue', ascending=False).head(20), x='warehouse_city', y='net_line_revenue', title='Revenue by Warehouse City') if 'warehouse_city' in f.columns else None),
        lambda: (px.bar(f.groupby('warehouse_state')['net_line_revenue'].sum().reset_index().sort_values('net_line_revenue', ascending=False).head(20), x='warehouse_state', y='net_line_revenue', title='Revenue by Warehouse State') if 'warehouse_state' in f.columns else None),
        lambda: chart_top_products(f, topn=20),
        lambda: chart_top_categories(f, topn=20),
        lambda: chart_margin_hist(f),
        lambda: chart_qty_distribution(f),
        lambda: chart_rev_distribution(f),
        lambda: (px.histogram(f.groupby('warehouse_name').agg(ship_share=('ship_share','sum')).reset_index()['ship_share'], nbins=40, title='Shipping share distribution by warehouse') if 'ship_share' in f.columns else None),
        lambda: (px.box(f, x='warehouse_name', y='net_line_revenue', title='Order revenue distribution by warehouse') if 'warehouse_name' in f.columns else None),
        lambda: (px.bar(f.groupby('warehouse_name')['qty'].sum().reset_index().sort_values('qty', ascending=False).head(20), x='warehouse_name', y='qty', title='Quantity sold by warehouse') if 'qty' in f.columns else None),
        lambda: chart_revenue_pareto(f),
        lambda: chart_top_customers(f, users, topn=10),
        lambda: chart_rev_vs_shipping(f),
        lambda: chart_discount_vs_rev(f),
        lambda: chart_top_products(f, topn=30),
        lambda: chart_top_categories(f, topn=30),
        lambda: (make_table_figure(warehouses[['warehouse_id','name','city','state']].rename(columns={'name':'warehouse_name'}) if isinstance(warehouses, pd.DataFrame) else pd.DataFrame())),
        lambda: chart_gp_by_warehouse(f),
        lambda: chart_rev_by_warehouse(f),
        lambda: chart_rev_distribution(f)
    ]
    return gens

# ---------- Render tabs by invoking generator arrays (up to 25 per tab) ----------
with tab_overview:
    st.header("Overview — 25 charts (generated)")
    gens = overview_generators()
    render_charts_for_tab(gens, tab_name="overview", max_charts=25)

with tab_products:
    st.header("Products & Category Profitability — 25 charts (generated)")
    gens = products_generators()
    render_charts_for_tab(gens, tab_name="products", max_charts=25)

with tab_customers:
    st.header("Customers — 25 charts (generated)")
    gens = customers_generators()
    render_charts_for_tab(gens, tab_name="customers", max_charts=25)

with tab_ops:
    st.header("Operations — 25 charts (generated)")
    gens = ops_generators()
    render_charts_for_tab(gens, tab_name="ops", max_charts=25)

with tab_marketing:
    st.header("Marketing — 25 charts (generated)")
    gens = marketing_generators()
    render_charts_for_tab(gens, tab_name="marketing", max_charts=25)

with tab_reviews:
    st.header("Reviews — 25 charts (generated)")
    gens = reviews_generators()
    render_charts_for_tab(gens, tab_name="reviews", max_charts=25)

with tab_warehouses:
    st.header("Warehouses — 25 charts (generated)")
    gens = warehouses_generators()
    render_charts_for_tab(gens, tab_name="warehouses", max_charts=25)

# ---------- Download & finishing ----------
if not f.empty:
    st.download_button("Download filtered fact CSV", data=f.to_csv(index=False), file_name="fact_snapshot.csv", mime="text/csv")
st.markdown("---")
st.caption("Generated up to 25 charts per tab. If you want specific custom charts or to reorder charts, tell me which tab and which charts to prioritize and I'll edit the exact sequence.")
