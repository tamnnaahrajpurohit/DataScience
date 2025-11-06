# app_olist.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Company Dashboard (Olist)", layout="wide")

@st.cache_data
def load_data():
    users = pd.read_csv("retail_dashboard/data/users.csv", parse_dates=["created_at"])
    warehouses = pd.read_csv("retail_dashboard/data/warehouses.csv")
    products = pd.read_csv("retail_dashboard/data/products.csv")
    orders = pd.read_csv("retail_dashboard/data/orders.csv", parse_dates=["order_date"])
    order_items = pd.read_csv("retail_dashboard/data/order_items.csv")
    reviews = pd.read_csv("retail_dashboard/data/reviews.csv", parse_dates=["review_date"])
    returns = pd.read_csv("retail_dashboard/data/returns.csv", parse_dates=["processed_date"])
    mkt = pd.read_csv("retail_dashboard/data/marketing_spend.csv")
    return users, warehouses, products, orders, order_items, reviews, returns, mkt

def build_fact_table(orders, order_items, products):
    df = order_items.copy()

    # ---- Fallbacks for RAW Olist columns ----
    # qty: raw Olist has 1 row per item; default to 1 if missing
    if 'qty' not in df.columns:
        df['qty'] = 1

    # unit_price fallback: use raw 'price' if present
    if 'unit_price' not in df.columns:
        if 'price' in df.columns:
            df['unit_price'] = pd.to_numeric(df['price'], errors='coerce')
        else:
            df['unit_price'] = np.nan  # will fill from products if available

    # orders: ensure order_date exists
    if 'order_date' not in orders.columns:
        # raw Olist name
        if 'order_purchase_timestamp' in orders.columns:
            orders = orders.copy()
            orders['order_date'] = pd.to_datetime(orders['order_purchase_timestamp'], errors='coerce')
        else:
            orders = orders.copy()
            orders['order_date'] = pd.NaT

    # shipping_cost / discount_amount fallbacks
    if 'shipping_cost' not in orders.columns:
        # try to compute from order_items.freight_value
        if 'freight_value' in order_items.columns:
            ship = order_items.groupby('order_id')['freight_value'].sum().rename('shipping_cost').reset_index()
            orders = orders.merge(ship, on='order_id', how='left')
        else:
            orders['shipping_cost'] = 0.0

    if 'discount_amount' not in orders.columns:
        # Approx: discount = max(0, (sum(price)+freight) - total payment)
        disc = None
        if 'price' in order_items.columns:
            line_sum = order_items.groupby('order_id')['price'].sum().rename('items_price')
            orders = orders.merge(line_sum, on='order_id', how='left')
        else:
            orders['items_price'] = 0.0

        if 'payment_value' in orders.columns:
            gross = orders['items_price'].fillna(0) + orders['shipping_cost'].fillna(0)
            orders['discount_amount'] = (gross - orders['payment_value'].fillna(gross)).clip(lower=0)
        else:
            orders['discount_amount'] = 0.0

    # warehouse_id fallback: raw Olist has no single warehouse per order
    if 'warehouse_id' not in orders.columns:
        if 'seller_id' in order_items.columns:
            first_seller = order_items.sort_values(['order_id','order_item_id']).drop_duplicates('order_id')[['order_id','seller_id']]
            first_seller = first_seller.rename(columns={'seller_id':'warehouse_id'})
            orders = orders.merge(first_seller, on='order_id', how='left')
        else:
            orders['warehouse_id'] = np.nan

    # channel fallback
    if 'channel' not in orders.columns:
        orders['channel'] = 'online'

    # Merge order columns
    orders_cols = ['order_id','order_date','discount_amount','shipping_cost','channel','warehouse_id']
    orders_cols = [c for c in orders_cols if c in orders.columns]
    df = df.merge(orders[orders_cols], on='order_id', how='left')

    # ---- Products enrichment (unit_cost / unit_price) ----
    prod = products.copy()

    # If products don't have unit_price/cost (raw Olist), derive from order_items
    if 'unit_price' not in prod.columns:
        if 'price' in order_items.columns:
            med_price = order_items.groupby('product_id')['price'].median().rename('unit_price')
            prod = prod.merge(med_price, on='product_id', how='left')
        else:
            prod['unit_price'] = np.nan
    if 'unit_cost' not in prod.columns:
        prod['unit_cost'] = prod['unit_price'] * 0.7  # assumption; swap to your COGS if you have it

    # SKU/category safe defaults
    if 'sku' not in prod.columns:
        prod['sku'] = prod.get('product_id', pd.Series(dtype=str)).astype(str).str[:8]
    if 'category' not in prod.columns:
        # raw Olist has product_category_name; keep that if present
        if 'product_category_name' in prod.columns:
            prod['category'] = prod['product_category_name']
        else:
            prod['category'] = 'unknown'

    prod_keep = ['product_id','sku','category','unit_cost','unit_price']
    prod_keep = [c for c in prod_keep if c in prod.columns]
    df = df.merge(prod[prod_keep].rename(columns={'unit_price':'unit_price_prod'}), on='product_id', how='left')

    # Prefer line unit_price; fill missing from product median price
    df['unit_price'] = df['unit_price'].fillna(df['unit_price_prod'])

    # ---- Metrics ----
    df['line_total'] = df['qty'] * df['unit_price']
    line_sum = df.groupby('order_id')['line_total'].transform('sum').replace(0, np.nan)
    df['discount_share'] = (df['line_total'] / line_sum) * df['discount_amount'].fillna(0)
    df['net_line_revenue'] = df['line_total'] - df['discount_share'].fillna(0)

    ship_sum = df.groupby('order_id')['line_total'].transform('sum').replace(0, np.nan)
    df['ship_share'] = (df['line_total'] / ship_sum) * df['shipping_cost'].fillna(0)

    df['cogs'] = (df['qty'] * df['unit_cost'].fillna(df['unit_price'] * 0.7)).fillna(0)

    df['gross_profit'] = df['net_line_revenue'] - df['cogs'] - df['ship_share']
    return df

users, warehouses, products, orders, order_items, reviews, returns, mkt = load_data()
fact = build_fact_table(orders, order_items, products)

st.sidebar.header("Filters")
min_date = orders["order_date"].min()
max_date = orders["order_date"].max()
date_range = st.sidebar.date_input("Order Date Range", value=(min_date, max_date))
channels = ["All"] + sorted(orders["channel"].dropna().unique().tolist())
channel_sel = st.sidebar.selectbox("Channel", channels, index=0)
whs = ["All"] + sorted(orders["warehouse_id"].dropna().astype(str).unique().tolist())
wh_sel = st.sidebar.selectbox("Warehouse (Seller)", whs, index=0)
cats = ["All"] + sorted(products["category"].dropna().unique().tolist())
cat_sel = st.sidebar.selectbox("Category", cats, index=0)

f = fact.copy()
if isinstance(date_range, (list, tuple)) and len(date_range)==2:
    s, e = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    f = f[(f["order_date"]>=s) & (f["order_date"]<=e)]
if channel_sel != "All":
    f = f[f["channel"]==channel_sel]
if wh_sel != "All":
    f = f[f["warehouse_id"].astype(str)==str(wh_sel)]
if cat_sel != "All":
    f = f[f["category"]==cat_sel]

st.title("Company Dashboard (Olist → Unified Schema)")

col1, col2, col3, col4 = st.columns(4)
total_rev = f["net_line_revenue"].sum()
gross_profit = f["gross_profit"].sum()
aov = f.groupby("order_id")["net_line_revenue"].sum().mean()
orders_count = f["order_id"].nunique()

col1.metric("Revenue (filtered)", f"R${total_rev:,.0f}")
col2.metric("Gross Profit", f"R${gross_profit:,.0f}")
col3.metric("AOV", f"R${aov:,.0f}" if not np.isnan(aov) else "N/A")
col4.metric("Orders", f"{orders_count:,}")

st.markdown("---")

st.subheader("Revenue Over Time (Weekly)")
rev_ts = f.groupby(pd.Grouper(key="order_date", freq="W"))["net_line_revenue"].sum().reset_index()
st.plotly_chart(px.line(rev_ts, x="order_date", y="net_line_revenue", markers=True), use_container_width=True)

st.subheader("Gross Profit by Category & Warehouse")
cat_wh = f.groupby(["category","warehouse_id"])["gross_profit"].sum().reset_index()
st.plotly_chart(px.bar(cat_wh, x="category", y="gross_profit", color="warehouse_id", barmode="group"), use_container_width=True)

st.subheader("Top Products (Revenue & Margin)")
prod_sum = f.groupby(["product_id","sku","category"]).agg(revenue=("net_line_revenue","sum"), margin=("gross_profit","sum"), qty=("qty","sum")).reset_index().sort_values("revenue", ascending=False).head(15)
st.plotly_chart(px.bar(prod_sum, x="sku", y=["revenue","margin"], barmode="group"), use_container_width=True)
st.dataframe(prod_sum)

st.markdown("---")

st.subheader("New Users by Month")
u = users.copy()
u["cohort_month"] = u["created_at"].dt.to_period("M").astype(str)
cohort = u.groupby("cohort_month")["user_id"].nunique().reset_index(name="new_users")
st.plotly_chart(px.bar(cohort, x="cohort_month", y="new_users"), use_container_width=True)

st.subheader("Reviews Overview")
if not reviews.empty:
    r = reviews.copy()
    r["review_month"] = pd.to_datetime(r["review_date"]).dt.to_period("M").astype(str)
    rating_dist = r["rating"].value_counts().sort_index()
    st.plotly_chart(px.bar(x=rating_dist.index, y=rating_dist.values, labels={"x":"Rating","y":"Count"}), use_container_width=True)
    avg_rating = r["rating"].mean()
    st.metric("Average Rating", f"{avg_rating:.2f}")
else:
    st.info("No reviews data found.")

st.markdown("---")

st.subheader("Marketing Spend & CPA (Synthetic)")
if not mkt.empty:
    f["month"] = f["order_date"].dt.to_period("M").astype(str)
    ord_by_chan = f.groupby(["month","channel"])["order_id"].nunique().reset_index(name="orders")
    mk = mkt.merge(ord_by_chan, on=["month","channel"], how="left").fillna({"orders":0})
    mk["cpa"] = mk["spend"] / mk["orders"].replace(0, np.nan)
    st.plotly_chart(px.bar(mk, x="month", y="spend", color="channel", barmode="group"), use_container_width=True)
    st.dataframe(mk[["month","channel","spend","orders","cpa"]])
else:
    st.info("No marketing spend data found.")


# -------------------------------
# ALL 25 VISUALS (single scroll)
# -------------------------------
def render_25_visuals(users, warehouses, products, orders, order_items, reviews, returns, fact):
    import re
    st.header("All 25 Visuals (Company-wide)")

    # Helper: safe count
    def nonempty(df): 
        return (df is not None) and (len(df) > 0)

    # Prepare common frames
    f = fact.copy()
    f["order_date"] = pd.to_datetime(f["order_date"], errors="coerce")
    f["month"] = f["order_date"].dt.to_period("M").astype(str)
    orders2 = orders.copy()
    if "order_date" not in orders2.columns and "order_purchase_timestamp" in orders2.columns:
        orders2["order_date"] = pd.to_datetime(orders2["order_purchase_timestamp"], errors="coerce")
    orders2["month"] = pd.to_datetime(orders2["order_date"], errors="coerce").dt.to_period("M").astype(str)

    # 01 Revenue distribution (hist)
    st.subheader("01 — Revenue per Order (Histogram)")
    rev_per_order = f.groupby("order_id")["net_line_revenue"].sum().reset_index()
    st.plotly_chart(px.histogram(rev_per_order, x="net_line_revenue", nbins=50), use_container_width=True)

    # 02 Orders by status
    st.subheader("02 — Orders by Status")
    if "status" in orders2.columns:
        s = orders2["status"].value_counts().reset_index()
        s.columns = ["status","count"]
        st.plotly_chart(px.bar(s, x="status", y="count"), use_container_width=True)

    # 03 AOV over time (monthly)
    st.subheader("03 — Average Order Value (Monthly)")
    aov_m = f.groupby("month")["net_line_revenue"].sum().reset_index()
    ord_m = f.groupby("month")["order_id"].nunique().reset_index()
    aov_m = aov_m.merge(ord_m, on="month", how="left")
    aov_m["AOV"] = aov_m["net_line_revenue"] / aov_m["order_id"]
    st.plotly_chart(px.line(aov_m, x="month", y="AOV", markers=True), use_container_width=True)

    # 04 Revenue by Category (Top 15)
    st.subheader("04 — Revenue by Category (Top 15)")
    cat_rev = f.groupby("category")["net_line_revenue"].sum().sort_values(ascending=False).head(15).reset_index()
    st.plotly_chart(px.bar(cat_rev, x="category", y="net_line_revenue"), use_container_width=True)

    # 05 Gross Profit by Category
    st.subheader("05 — Gross Profit by Category")
    cat_gp = f.groupby("category")["gross_profit"].sum().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.bar(cat_gp, x="category", y="gross_profit"), use_container_width=True)

    # 06 Revenue by Warehouse
    st.subheader("06 — Revenue by Warehouse (Seller)")
    wh_rev = f.groupby("warehouse_id")["net_line_revenue"].sum().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.bar(wh_rev, x="warehouse_id", y="net_line_revenue"), use_container_width=True)

    # 07 Gross Profit by Warehouse
    st.subheader("07 — Gross Profit by Warehouse")
    wh_gp = f.groupby("warehouse_id")["gross_profit"].sum().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.bar(wh_gp, x="warehouse_id", y="gross_profit"), use_container_width=True)

    # 08 Channel Mix (Revenue)
    st.subheader("08 — Revenue by Channel")
    if "channel" in f.columns:
        ch = f.groupby("channel")["net_line_revenue"].sum().reset_index()
        st.plotly_chart(px.bar(ch, x="channel", y="net_line_revenue"), use_container_width=True)

    # # 09 Weekly Revenue Trend
    # st.subheader("09 — Weekly Revenue Trend")
    # weekly = f.groupby(pd.Grouper(key="order_date", freq="W"))["net_line_revenue"].sum().reset_index()
    # st.plotly_chart(px.line(weekly, x="order_date", y="net_line_revenue", markers=True), use_container_width=True)

    # 10 Monthly Gross Profit Trend
    st.subheader("10 — Monthly Gross Profit Trend")
    gp_m = f.groupby("month")["gross_profit"].sum().reset_index()
    st.plotly_chart(px.line(gp_m, x="month", y="gross_profit", markers=True), use_container_width=True)

    # 11 Top Products by Revenue
    st.subheader("11 — Top Products by Revenue (Top 15)")
    top_prod_rev = f.groupby(["product_id","sku"]).agg(revenue=("net_line_revenue","sum")).reset_index().sort_values("revenue", ascending=False).head(15)
    st.plotly_chart(px.bar(top_prod_rev, x="sku", y="revenue"), use_container_width=True)

    # 12 Top Products by Margin
    st.subheader("12 — Top Products by Margin (Top 15)")
    top_prod_mg = f.groupby(["product_id","sku"]).agg(margin=("gross_profit","sum")).reset_index().sort_values("margin", ascending=False).head(15)
    st.plotly_chart(px.bar(top_prod_mg, x="sku", y="margin"), use_container_width=True)

    # 13 Popularity vs Quality (Qty vs Avg Margin)
    st.subheader("13 — Popularity vs Profitability")
    pq = f.groupby(["product_id","sku"]).agg(qty=("qty","sum"), avg_margin=("gross_profit","mean")).reset_index()
    st.plotly_chart(px.scatter(pq, x="qty", y="avg_margin", hover_data=["sku"]), use_container_width=True)

    # 14 Price vs Unit Cost (Products)
    st.subheader("14 — Price vs Unit Cost (Products)")
    p2 = products.copy()
    st.plotly_chart(px.scatter(p2, x="unit_price", y="unit_cost", hover_data=["sku","category"]), use_container_width=True)

    # 15 Category Share (Revenue Pie)
    st.subheader("15 — Category Revenue Share")
    st.plotly_chart(px.pie(cat_rev, names="category", values="net_line_revenue"), use_container_width=True)

    # 16 Return Rate by Month
    st.subheader("16 — Return/Cancel Rate by Month")
    o = orders2.copy()
    if "status" in o.columns:
        tot = o.groupby("month")["order_id"].nunique().rename("orders_total")
        ret = o[o["status"].isin(["canceled","unavailable"])].groupby("month")["order_id"].nunique().rename("orders_returned")
        rr = pd.concat([tot, ret], axis=1).fillna(0).reset_index()
        rr["return_rate"] = (rr["orders_returned"] / rr["orders_total"]).replace([np.inf, np.nan], 0)
        st.plotly_chart(px.line(rr, x="month", y="return_rate", markers=True), use_container_width=True)

    # 17 Rating Distribution
    st.subheader("17 — Rating Distribution")
    if nonempty(reviews) and "rating" in reviews.columns:
        rd = reviews["rating"].value_counts().sort_index().reset_index()
        rd.columns = ["rating","count"]
        st.plotly_chart(px.bar(rd, x="rating", y="count"), use_container_width=True)

    # 18 Reviews over Time
    st.subheader("18 — Reviews per Month")
    if nonempty(reviews) and "review_date" in reviews.columns:
        r2 = reviews.copy()
        r2["rm"] = pd.to_datetime(r2["review_date"], errors="coerce").dt.to_period("M").astype(str)
        rpm = r2.groupby("rm")["review_id"].nunique().reset_index(name="reviews")
        st.plotly_chart(px.line(rpm, x="rm", y="reviews", markers=True), use_container_width=True)

    # 19 Return Rate vs Rating (join reviews to orders)
    st.subheader("19 — Return Rate vs Rating")
    if nonempty(reviews) and "status" in orders2.columns:
        orr = orders2[["order_id","status"]].copy()
        orr["is_returned"] = orr["status"].isin(["canceled","unavailable"]).astype(int)
        rj = reviews.merge(orr, on="order_id", how="left")
        byrat = rj.groupby("rating")["is_returned"].mean().reset_index()
        st.plotly_chart(px.line(byrat, x="rating", y="is_returned", markers=True), use_container_width=True)

    # 20 City-wise Users
    st.subheader("20 — Top Cities by Users")
    if nonempty(users):
        uc = users["city"].value_counts().head(20).reset_index()
        uc.columns = ["city","users"]
        st.plotly_chart(px.bar(uc, x="city", y="users"), use_container_width=True)

    # 21 State-wise Orders
    st.subheader("21 — Orders by Customer State")
    if "customer_state" in orders.columns:
        st_state = orders["customer_state"].value_counts().reset_index()
        st_state.columns = ["state","orders"]
        st.plotly_chart(px.bar(st_state, x="state", y="orders"), use_container_width=True)

    # 22 Shipping Cost vs Revenue (order-level)
    st.subheader("22 — Shipping Cost vs Order Revenue")
    ord_rev = f.groupby("order_id")[["net_line_revenue","ship_share"]].sum().reset_index()
    ord_rev["shipping_cost"] = ord_rev["ship_share"]
    st.plotly_chart(px.scatter(ord_rev, x="net_line_revenue", y="shipping_cost"), use_container_width=True)

    # 23 Discount vs Revenue (order-level)
    st.subheader("23 — Discount vs Order Revenue")
    disc_ord = f.groupby("order_id")[["net_line_revenue","discount_share"]].sum().reset_index()
    disc_ord["discount_amount"] = disc_ord["discount_share"]
    st.plotly_chart(px.scatter(disc_ord, x="net_line_revenue", y="discount_amount"), use_container_width=True)

    # 24 Profit Waterfall by Month (bar: revenue & margin)
    st.subheader("24 — Revenue vs Gross Profit (Monthly)")
    rev_m = f.groupby("month")["net_line_revenue"].sum().rename("revenue").reset_index()
    mg_m = f.groupby("month")["gross_profit"].sum().rename("gross_profit").reset_index()
    both = rev_m.merge(mg_m, on="month", how="left")
    st.plotly_chart(px.bar(both, x="month", y=["revenue","gross_profit"], barmode="group"), use_container_width=True)

    # 25 Category Profitability Scatter (Rev vs Margin)
    st.subheader("25 — Category-level Revenue vs Margin")
    cat_sc = f.groupby("category").agg(revenue=("net_line_revenue","sum"), margin=("gross_profit","sum")).reset_index()
    st.plotly_chart(px.scatter(cat_sc, x="revenue", y="margin", hover_data=["category"]), use_container_width=True)


render_25_visuals(users, warehouses, products, orders, order_items, reviews, returns, fact)
