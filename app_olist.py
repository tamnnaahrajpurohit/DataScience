# app_olist.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine

def mysql_query(query):
    # Use SQLAlchemy to avoid pandas warning
    # Try pymysql first (most common), fallback to mysqlconnector
    try:
        import pymysql
        connection_string = "mysql+pymysql://root:@localhost/ecommer_python_datascience"
    except ImportError:
        # Fallback to mysqlconnector if pymysql is not available
        connection_string = "mysql+mysqlconnector://root:@localhost/ecommer_python_datascience"
    engine = create_engine(connection_string)
    df = pd.read_sql(query, engine)
    engine.dispose()
    return df
st.set_page_config(page_title="E-Commerce Analytics Dashboard for Profitability & Operations (Olist) — Insights & Profitability", layout="wide")

def read_csv_optional(path, parse_dates=None):
    if path is None:
        return pd.DataFrame()
    try:
        if os.path.exists(path):
            header = pd.read_csv(path, nrows=0)
            if parse_dates:
                cols_to_parse = [c for c in (parse_dates if isinstance(parse_dates, (list,tuple)) else [parse_dates]) if c in header.columns]
                if cols_to_parse:
                    return pd.read_csv(path, parse_dates=cols_to_parse)
                else:
                    return pd.read_csv(path)
            else:
                return pd.read_csv(path)
        else:
            return pd.DataFrame()
    except Exception:
        try:
            if os.path.exists(path):
                return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data
def load_data():
    base = "retail_dashboard/data"
    users = mysql_query("SELECT * FROM users")
    warehouses = mysql_query("SELECT * FROM warehouses")
    products = mysql_query("SELECT * FROM products")
    orders = mysql_query("SELECT * FROM orders")
    order_items = mysql_query("SELECT * FROM order_items")
    reviews = mysql_query("SELECT * FROM reviews")
    returns = mysql_query("SELECT * FROM returns")
    mkt = mysql_query("SELECT * FROM marketing_spend")
    # users = read_csv_optional(os.path.join(base, "users.csv"), parse_dates=["created_at"])
    # warehouses = read_csv_optional(os.path.join(base, "warehouses.csv"))
    # products = read_csv_optional(os.path.join(base, "products.csv"))
    # orders = read_csv_optional(os.path.join(base, "orders.csv"))
    # order_items = read_csv_optional(os.path.join(base, "order_items.csv"))
    # reviews = read_csv_optional(os.path.join(base, "reviews.csv"), parse_dates=["review_date"])
    # returns = read_csv_optional(os.path.join(base, "returns.csv"), parse_dates=["processed_date"])
    # mkt = read_csv_optional(os.path.join(base, "marketing_spend.csv"))
    return users, warehouses, products, orders, order_items, reviews, returns, mkt
def beautify_label(label):
    """Convert snake_case or column names to Title Case labels"""
    try:
        if label is None:
            return ""
        if not isinstance(label, str):
            return str(label)
        return label.replace("_", " ").title()
    except Exception:
        return str(label)

def beautify_plot(fig):
    """Axis Titles."""
    try:
        # Title
        title = fig.layout.title.text if hasattr(fig.layout, "title") and fig.layout.title.text is not None else ""
        fig.update_layout(title=beautify_label(title))

        # xaxis
        if hasattr(fig.layout, "xaxis") and fig.layout.xaxis.title.text is not None:
            fig.update_layout(xaxis_title=beautify_label(fig.layout.xaxis.title.text))
        # yaxis
        if hasattr(fig.layout, "yaxis") and fig.layout.yaxis.title.text is not None:
            fig.update_layout(yaxis_title=beautify_label(fig.layout.yaxis.title.text))
    except Exception:
        pass
    return fig

def build_fact_table(orders, order_items, products):
    df = order_items.copy() if isinstance(order_items, pd.DataFrame) else pd.DataFrame()
    if df.empty:
        return df

    # Ensure qty is numeric
    if 'qty' not in df.columns:
        df['qty'] = 1
    else:
        df['qty'] = pd.to_numeric(df['qty'], errors='coerce').fillna(1)

    # Ensure unit_price is numeric
    if 'unit_price' not in df.columns:
        if 'price' in df.columns:
            df['unit_price'] = pd.to_numeric(df['price'], errors='coerce')
        else:
            df['unit_price'] = np.nan
    else:
        df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce')

    orders = orders.copy() if isinstance(orders, pd.DataFrame) else pd.DataFrame()
    if 'order_date' not in orders.columns:
        if 'order_purchase_timestamp' in orders.columns:
            try:
                orders['order_date'] = pd.to_datetime(orders['order_purchase_timestamp'], errors='coerce')
            except Exception:
                orders['order_date'] = pd.NaT
        else:
            orders['order_date'] = pd.NaT

    if 'shipping_cost' not in orders.columns:
        if 'freight_value' in df.columns:
            # Ensure freight_value is numeric before summing
            df['freight_value'] = pd.to_numeric(df['freight_value'], errors='coerce').fillna(0)
            ship = df.groupby('order_id')['freight_value'].sum().rename('shipping_cost').reset_index()
            orders = orders.merge(ship, on='order_id', how='left')
        else:
            orders['shipping_cost'] = 0.0

    if 'discount_amount' not in orders.columns:
        if 'price' in df.columns:
            # Ensure price is numeric before summing
            df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
            line_sum = df.groupby('order_id')['price'].sum().rename('items_price').reset_index()
            orders = orders.merge(line_sum, on='order_id', how='left')
        else:
            orders['items_price'] = 0.0

        if 'payment_value' in orders.columns:
            # Ensure all values are numeric before calculation
            orders['items_price'] = pd.to_numeric(orders['items_price'], errors='coerce').fillna(0)
            orders['shipping_cost'] = pd.to_numeric(orders['shipping_cost'], errors='coerce').fillna(0)
            orders['payment_value'] = pd.to_numeric(orders['payment_value'], errors='coerce').fillna(0)
            gross = orders['items_price'] + orders['shipping_cost']
            orders['discount_amount'] = (gross - orders['payment_value']).clip(lower=0)
        else:
            orders['discount_amount'] = 0.0

    if 'warehouse_id' not in orders.columns:
        if 'seller_id' in df.columns:
            order_item_col = 'order_item_id' if 'order_item_id' in df.columns else df.columns[0]
            first_seller = df.sort_values(['order_id', order_item_col]).drop_duplicates('order_id')[['order_id','seller_id']].rename(columns={'seller_id':'warehouse_id'})
            orders = orders.merge(first_seller, on='order_id', how='left')
        else:
            orders['warehouse_id'] = np.nan

    if 'channel' not in orders.columns:
        orders['channel'] = 'online'

    orders_cols = ['order_id','order_date','discount_amount','shipping_cost','channel','warehouse_id']
    orders_cols = [c for c in orders_cols if c in orders.columns]
    df = df.merge(orders[orders_cols], on='order_id', how='left')
    
    # Ensure numeric columns from orders merge are numeric
    if 'discount_amount' in df.columns:
        df['discount_amount'] = pd.to_numeric(df['discount_amount'], errors='coerce').fillna(0)
    if 'shipping_cost' in df.columns:
        df['shipping_cost'] = pd.to_numeric(df['shipping_cost'], errors='coerce').fillna(0)

    prod = products.copy() if isinstance(products, pd.DataFrame) else pd.DataFrame()
    if prod.empty:
        prod = df[['product_id']].drop_duplicates().assign(unit_price=np.nan, unit_cost=np.nan, sku=df.get('product_id', pd.Series()).astype(str).str[:8], category='unknown')

    if 'unit_price' not in prod.columns:
        if 'price' in df.columns:
            # Ensure price is numeric before calculating median
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            med_price = df.groupby('product_id')['price'].median().rename('unit_price').reset_index()
            prod = prod.merge(med_price, on='product_id', how='left')
        else:
            prod['unit_price'] = np.nan

    if 'unit_cost' not in prod.columns:
        # Ensure unit_price is numeric before calculation
        prod['unit_price'] = pd.to_numeric(prod['unit_price'], errors='coerce')
        prod['unit_cost'] = prod['unit_price'].fillna(0) * 0.7
    
    # Ensure unit_price and unit_cost are numeric in products dataframe
    if 'unit_price' in prod.columns:
        prod['unit_price'] = pd.to_numeric(prod['unit_price'], errors='coerce')
    if 'unit_cost' in prod.columns:
        prod['unit_cost'] = pd.to_numeric(prod['unit_cost'], errors='coerce')

    if 'sku' not in prod.columns:
        prod['sku'] = prod.get('product_id', pd.Series(dtype=str)).astype(str).str[:8]

    if 'category' not in prod.columns:
        if 'product_category_name' in prod.columns:
            prod['category'] = prod['product_category_name']
        else:
            prod['category'] = prod.get('category', 'unknown')

    prod_keep = ['product_id','sku','category','unit_cost','unit_price']
    prod_keep = [c for c in prod_keep if c in prod.columns]
    df = df.merge(prod[prod_keep].rename(columns={'unit_price':'unit_price_prod'}), on='product_id', how='left')

    df['unit_price'] = df['unit_price'].fillna(df.get('unit_price_prod', np.nan))
    # Ensure both columns are numeric before multiplication
    df['qty'] = pd.to_numeric(df['qty'], errors='coerce').fillna(1)
    df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce')
    
    # Ensure unit_cost is numeric if it exists
    if 'unit_cost' in df.columns:
        df['unit_cost'] = pd.to_numeric(df['unit_cost'], errors='coerce')
    
    df['line_total'] = df['qty'] * df['unit_price']
    line_sum = df.groupby('order_id')['line_total'].transform('sum').replace(0, np.nan)
    
    # Ensure discount_amount and shipping_cost are numeric before calculations
    if 'discount_amount' in df.columns:
        df['discount_amount'] = pd.to_numeric(df['discount_amount'], errors='coerce').fillna(0)
    else:
        df['discount_amount'] = 0.0
    
    if 'shipping_cost' in df.columns:
        df['shipping_cost'] = pd.to_numeric(df['shipping_cost'], errors='coerce').fillna(0)
    else:
        df['shipping_cost'] = 0.0
    
    df['discount_share'] = (df['line_total'] / line_sum) * df['discount_amount']
    df['net_line_revenue'] = df['line_total'] - df['discount_share'].fillna(0)
    ship_sum = df.groupby('order_id')['line_total'].transform('sum').replace(0, np.nan)
    df['ship_share'] = (df['line_total'] / ship_sum) * df['shipping_cost']
    df['cogs'] = (df['qty'] * df['unit_cost'].fillna(df['unit_price'] * 0.7)).fillna(0)
    df['gross_profit'] = df['net_line_revenue'] - df['cogs'] - df['ship_share']
    df['margin_pct'] = np.where(df['net_line_revenue']!=0, df['gross_profit']/df['net_line_revenue'], 0)
    return df

users, warehouses, products, orders, order_items, reviews, returns, mkt = load_data()
fact = build_fact_table(orders, order_items, products)

st.sidebar.header("Filters")
if isinstance(orders, pd.DataFrame) and 'order_date' in orders.columns and not orders['order_date'].isna().all():
    try:
        min_date = pd.to_datetime(orders["order_date"]).min()
        max_date = pd.to_datetime(orders["order_date"]).max()
    except Exception:
        min_date, max_date = pd.to_datetime("2000-01-01"), pd.to_datetime("2100-01-01")
else:
    min_date, max_date = pd.to_datetime("2000-01-01"), pd.to_datetime("2100-01-01")

date_range = st.sidebar.date_input("Order Date Range", value=(min_date, max_date))

if isinstance(orders, pd.DataFrame) and "channel" in orders.columns:
    channels = ["All"] + sorted(orders["channel"].dropna().unique().tolist())
else:
    channels = ["All"]
channel_sel = st.sidebar.selectbox("Channel", channels, index=0)

if (isinstance(orders, pd.DataFrame) and "warehouse_id" in orders.columns and isinstance(warehouses, pd.DataFrame) and {"warehouse_id", "warehouse_name"}.issubset(warehouses.columns)):
    orders_with_names = orders.merge(warehouses[["warehouse_id", "warehouse_name"]].drop_duplicates(),on="warehouse_id",how="left")
    wh_name_map = dict(zip(orders_with_names["warehouse_name"], orders_with_names["warehouse_id"]))
    wh_display = ["All"] + sorted(orders_with_names["warehouse_name"].dropna().unique().tolist())
    wh_sel_name = st.sidebar.selectbox("Warehouse (Seller)", wh_display, index=0)
    wh_sel_id = wh_name_map.get(wh_sel_name) if wh_sel_name != "All" else None
    wh_sel = wh_sel_id if wh_sel_id is not None else "All"
else:
    wh_display = ["All"]
    wh_sel_name = st.sidebar.selectbox("Warehouse (Seller)", wh_display, index=0)
    wh_sel_id = None
    wh_sel = "All"
if wh_sel_id:
    f = f[f["warehouse_id"] == wh_sel_id]


if isinstance(products, pd.DataFrame) and "category" in products.columns:
    cats = ["All"] + sorted(products["category"].dropna().unique().tolist())
else:
    cats = ["All"]
cat_sel = st.sidebar.selectbox("Category", cats, index=0)

f = fact.copy() if isinstance(fact, pd.DataFrame) else pd.DataFrame()
if not f.empty:
    if isinstance(date_range, (list, tuple)) and len(date_range)==2:
        s, e = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        f = f[(pd.to_datetime(f["order_date"], errors='coerce')>=s) & (pd.to_datetime(f["order_date"], errors='coerce')<=e)]
    if channel_sel != "All" and "channel" in f.columns:
        f = f[f["channel"]==channel_sel]
    if wh_sel != "All" and "warehouse_id" in f.columns:
        f = f[f["warehouse_id"].astype(str)==str(wh_sel)]
    if cat_sel != "All" and "category" in f.columns:
        f = f[f["category"]==cat_sel]
st.sidebar.markdown("---")
st.sidebar.caption("Note: Adjust filters to focus your analysis — charts and insights update in real time.")
def money(amount):
    """
    Format a numeric amount into a compact readable format:
    100 -> 100
    1,000 -> 1K
    1,20,000 -> 1.2L
    1,00,00,000 -> 1Cr
    Handles negatives and missing values gracefully.
    """
    try:
        amount = float(amount)
    except (ValueError, TypeError):
        return "0"

    # Handle negative numbers
    sign = "-" if amount < 0 else ""
    amount = abs(amount)

    if amount < 1_000:
        formatted = f"{amount:,.0f}"  # e.g. 500, 999
    elif amount < 1_00_000:
        formatted = f"{amount / 1_000:.1f}K"  # e.g. 12.3K
    elif amount < 1_00_00_000:
        formatted = f"{amount / 1_00_000:.1f}L"  # e.g. 4.5L
    else:
        formatted = f"{amount / 1_00_00_000:.1f}Cr"  # e.g. 2.1Cr

    # Remove trailing .0 (like 1.0K → 1K)
    if formatted.endswith(".0K") or formatted.endswith(".0L") or formatted.endswith(".0Cr"):
        formatted = formatted.replace(".0", "")

    return f"{sign}{formatted}"

st.title("Dashboard — Insights & Profitability")

if f.empty:
    st.warning("No Data found Check CSV files in `retail_dashboard/data/`.")
else:
    total_rev = f["net_line_revenue"].sum()
    gross_profit = f["gross_profit"].sum()
    orders_count = f["order_id"].nunique()
    aov = f.groupby("order_id")["net_line_revenue"].sum().mean()
    margin_pct = (gross_profit / total_rev) if total_rev!=0 else 0

    rr = 0.0
    if isinstance(orders, pd.DataFrame) and "status" in orders.columns:
        o2 = orders.copy()
        o2['is_returned'] = o2['status'].isin(["canceled","unavailable"]).astype(int)
        tot = o2['order_id'].nunique()
        ret = o2[o2['is_returned']==1]['order_id'].nunique()
        rr = (ret / tot) if tot>0 else 0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Revenue (filtered)", f"${money(total_rev)}")
    k2.metric("Gross Profit", money(gross_profit))
    k3.metric("Gross Margin %", f"{margin_pct*100:.2f}%")
    k4.metric("Orders", money(orders_count))
    k5.metric("AOV", f"${money(aov)}" if not np.isnan(aov) else "N/A")
    k6.metric("Return Rate", f"{rr*100:.2f}%")

st.markdown("---")

tab_overview, tab_products, tab_customers, tab_ops, tab_marketing, tab_reviews = st.tabs(["Overview", "Products & Categories", "Customers", "Operations", "Marketing", "Reviews & Rating"])

with tab_overview:
    st.header("Overview — Revenue & Profitability Trends")
    if not f.empty:
        f["order_date"] = pd.to_datetime(f["order_date"], errors="coerce")
        f["week"] = f["order_date"].dt.to_period("W").astype(str)
        f["month"] = f["order_date"].dt.to_period("M").astype(str)

        rev_ts = f.groupby(pd.Grouper(key="order_date", freq="W"))["net_line_revenue"].sum().reset_index().rename(columns={"order_date": "Order Date", "net_line_revenue": "Net Line Revenue"})
        fig = px.line(rev_ts, x="Order Date", y="Net Line Revenue", markers=True, title="Revenue (Weekly)")
        fig.update_layout(yaxis_tickprefix="$")
        beautify_plot(fig)
        st.plotly_chart(fig, use_container_width=True, key="plot_01")

        rev_m = f.groupby("month")["net_line_revenue"].sum().rename("Revenue").reset_index()
        gp_m = f.groupby("month")["gross_profit"].sum().rename("Gross Profit").reset_index()

        both = rev_m.merge(gp_m, on="month", how="left").fillna(0)
        both["Order Date"] = both["month"]

        fig2 = px.bar(
            both,
            x="Order Date",
            y=["Revenue", "Gross Profit"],
            barmode="group",
            title="Revenue vs Gross Profit (Monthly)"
        )

        fig2.update_layout(yaxis_tickprefix="$")
        beautify_plot(fig2)
        st.plotly_chart(fig2, use_container_width=True, key="plot_02")


        col1, col2 = st.columns(2)
        with col1:
            rev_per_order = f.groupby("order_id")["net_line_revenue"].sum().reset_index()
            fig3 = px.histogram(rev_per_order, x="net_line_revenue", nbins=60, title="Revenue per Order Distribution")
            fig3.update_layout(yaxis_title="Count", xaxis_title="Revenue per Order", xaxis_tickprefix="$")
            beautify_plot(fig3)
            st.plotly_chart(fig3, use_container_width=True, key="plot_03")
        with col2:
            aov_by_month = f.groupby("month").apply(lambda x: x.groupby("order_id")["net_line_revenue"].sum().mean()).reset_index(name="AOV")
            fig4 = px.line(aov_by_month, x="month", y="AOV", markers=True, title="AOV (Monthly)")
            beautify_plot(fig4)
            st.plotly_chart(fig4, use_container_width=True, key="plot_04")

        kp = both.tail(12).melt(id_vars="month", value_vars=["Revenue","Gross Profit"], var_name="metric", value_name="value")
        fig_kp = px.area(kp, x="month", y="value", color="metric", facet_col="metric", title="Last 12 Months: Revenue & Gross Profit (area)")
        beautify_plot(fig_kp)
        st.plotly_chart(fig_kp, use_container_width=True, key="plot_05")
    else:
        st.info("No transactional data available for overview.")

with tab_products:
    st.header("Products & Category Profitability")

    if not f.empty:
        prod = f.copy()

        # ✅ Ensure product_name exists by merging with products.csv
        if 'product_id' in prod.columns and isinstance(products, pd.DataFrame) and not products.empty:
            prod = prod.merge(
                products[['product_id', 'product_name'] + 
                         [c for c in ['sku', 'category'] if c in products.columns]],
                on='product_id',
                how='left'
            )

        # ✅ Fallback if no product_name available
        if 'product_name' not in prod.columns or prod['product_name'].isnull().all():
            prod['product_name'] = prod['product_id'].astype(str)

        # ✅ Dynamically determine grouping columns
        group_cols = ['product_name']
        if 'sku' in prod.columns:
            group_cols.append('sku')
        if 'category' in prod.columns:
            group_cols.append('category')

        # ✅ Aggregate revenue, profit, qty, margin
        prod_sum = prod.groupby(group_cols, dropna=False).agg(
            revenue=("net_line_revenue", "sum"),
            margin=("gross_profit", "sum"),
            qty=("qty", "sum"),
            margin_pct=("margin_pct", "mean")
        ).reset_index().sort_values("revenue", ascending=False)

        # ---------- Top 15 Products ----------
        st.subheader("Top 15 Products by Revenue")
        top15 = prod_sum.head(15)
        fig_p = px.bar(
            top15,
            x="product_name",
            y="revenue",
            hover_data=[col for col in ["margin", "qty"] if col in top15.columns],
            title="Top 15 Products (Revenue)"
        )
        fig_p.update_layout(
            yaxis_tickprefix="₹",
            xaxis_title="Product Name",
            yaxis_title="Revenue"
        )
        beautify_plot(fig_p)
        st.plotly_chart(fig_p, use_container_width=True, key="plot_06")

        st.dataframe(
            top15.style.format({
                "revenue": "{:.0f}",
                "margin": "{:.0f}",
                "margin_pct": "{:.2%}"
            }),
            height=300
        )

        # ---------- Category Profitability ----------
        if 'category' in f.columns:
            st.subheader("Category Profitability (Revenue vs Margin)")
            cat_sc = f.groupby("category").agg(
                revenue=("net_line_revenue", "sum"),
                margin=("gross_profit", "sum")
            ).reset_index()
            fig_cat = px.scatter(
                cat_sc,
                x="revenue",
                y="margin",
                size="revenue",
                hover_data=["category"],
                title="Category: Revenue vs Margin"
            )
            fig_cat.update_layout(
                yaxis_tickprefix="₹",
                xaxis_tickprefix="₹",
                xaxis_title="Revenue",
                yaxis_title="Margin"
            )
            beautify_plot(fig_cat)
            st.plotly_chart(fig_cat, use_container_width=True, key="plot_07")
        else:
            st.info("Category data unavailable to show category profitability.")

        # ---------- Loss-Making Products ----------
        st.subheader("Products Losing Money (Negative Gross Profit)")
        loss_prods = prod_sum[prod_sum["margin"] < 0].sort_values("margin").head(20)
        if not loss_prods.empty:
            fig_loss = px.bar(
                loss_prods,
                x="product_name",
                y="margin",
                title="Top Loss-Making SKUs"
            )
            fig_loss.update_layout(
                yaxis_tickprefix="₹",
                xaxis_title="Product Name",
                yaxis_title="Gross Profit"
            )
            beautify_plot(fig_loss)
            st.plotly_chart(fig_loss, use_container_width=True, key="plot_08")
            st.dataframe(loss_prods.style.format({"revenue": "{:.0f}", "margin": "{:.0f}"}))
        else:
            st.info("No negative-margin products in filtered data.")

        # ---------- Unit Price vs Unit Cost ----------
        st.subheader("Unit Price vs Unit Cost (Products)")
        p2 = products.copy() if isinstance(products, pd.DataFrame) and not products.empty else pd.DataFrame()
        if not p2.empty and {"unit_price", "unit_cost"}.issubset(p2.columns):
            fig_price_cost = px.scatter(
                p2,
                x="unit_price",
                y="unit_cost",
                hover_data=[col for col in ["sku", "category", "product_name"] if col in p2.columns],
                title="Unit Price vs Unit Cost"
            )
            beautify_plot(fig_price_cost)
            st.plotly_chart(fig_price_cost, use_container_width=True, key="plot_09")
        else:
            st.info("Product price/cost data missing to show Price vs Cost chart.")

    else:
        st.info("No product transactions to analyze.")

with tab_customers:
    st.header("Customer Insights & Retention")
    if isinstance(users, pd.DataFrame) and not users.empty and not f.empty:
        u = users.copy()
        if 'created_at' in u.columns:
            u["cohort_month"] = pd.to_datetime(u["created_at"], errors="coerce").dt.to_period("M").astype(str)
            cohort = u.groupby("cohort_month")["user_id"].nunique().reset_index(name="new_users")
            st.subheader("New Users by Month")
            st.plotly_chart(px.bar(cohort, x="cohort_month", y="new_users", title="New users per month"), use_container_width=True, key="plot_10")
        else:
            st.info("created_at column missing in users to compute cohorts.")

        orders_by_user = orders.copy() if isinstance(orders, pd.DataFrame) else pd.DataFrame()
        if not orders_by_user.empty and "customer_id" in orders_by_user.columns:
            ob = orders_by_user.groupby("customer_id")["order_id"].nunique().reset_index(name="orders_per_user")
            repeat_rate = (ob["orders_per_user"]>1).mean()
            st.metric("Repeat Purchase Rate", f"{repeat_rate*100:.2f}%")
            st.subheader("Top Cities by Users")
            if "city" in users.columns:
                uc = users["city"].value_counts().head(20).reset_index()
                uc.columns = ["city","users"]
                st.plotly_chart(px.bar(uc, x="city", y="users", title="Top Cities by Registered Users"), use_container_width=True, key="plot_11")
            else:
                st.info("City column missing in users data.")
        else:
            st.info("Customer-level order data missing to compute repeat purchase rate.")
    else:
        st.info("User or transaction data missing for customer insights.")

with tab_ops:
    st.header("Operations — Returns, Shipping & Discounts")
    orders2 = orders.copy() if isinstance(orders, pd.DataFrame) else pd.DataFrame()
    if "order_date" not in orders2.columns and "order_purchase_timestamp" in orders2.columns:
        orders2["order_date"] = pd.to_datetime(orders2["order_purchase_timestamp"], errors="coerce")
    if not f.empty:
        if "status" in orders2.columns:
            orders2["month"] = pd.to_datetime(orders2["order_date"], errors="coerce").dt.to_period("M").astype(str)
            tot = orders2.groupby("month")["order_id"].nunique().rename("orders_total")
            ret = orders2[orders2["status"].isin(["canceled","unavailable"])].groupby("month")["order_id"].nunique().rename("orders_returned")
            rr_df = pd.concat([tot, ret], axis=1).fillna(0).reset_index()
            rr_df["return_rate"] = (rr_df["orders_returned"] / rr_df["orders_total"]).replace([np.inf, np.nan], 0)
            st.subheader("Return / Cancel Rate by Month")
            st.plotly_chart(px.line(rr_df, x="month", y="return_rate", markers=True, title="Return Rate by Month"), use_container_width=True, key="plot_12")

        st.subheader("Shipping Cost vs Order Revenue")
        ord_rev = f.groupby("order_id")[["net_line_revenue","ship_share","discount_share"]].sum().reset_index()
        ord_rev = ord_rev.rename(columns={"ship_share":"shipping_cost","discount_share":"discount_amount"})
        fig_ship = px.scatter(ord_rev, x="net_line_revenue", y="shipping_cost", title="Shipping vs Order Revenue", hover_data=["discount_amount"])
        fig_ship.update_layout(yaxis_tickprefix="$", xaxis_tickprefix="$")
        st.plotly_chart(fig_ship, use_container_width=True, key="plot_13")

        st.subheader("Discount Impact: Discount vs Revenue per Order")
        fig_disc = px.scatter(ord_rev, x="net_line_revenue", y="discount_amount", title="Discount vs Order Revenue")
        st.plotly_chart(fig_disc, use_container_width=True, key="plot_14")
    else:
        st.info("No transactional data for operations metrics.")

with tab_marketing:
    st.header("Marketing — Spend & Synthetic CPA")
    if isinstance(mkt, pd.DataFrame) and not mkt.empty and not f.empty:
        f["month"] = f["order_date"].dt.to_period("M").astype(str)
        ord_by_chan = f.groupby(["month","channel"])["order_id"].nunique().reset_index(name="orders")
        mk = mkt.merge(ord_by_chan, on=["month","channel"], how="left").fillna({"orders":0})
        mk["cpa"] = mk["spend"] / mk["orders"].replace(0, np.nan)
        st.subheader("Marketing Spend by Channel (Monthly)")
        st.plotly_chart(px.bar(mk, x="month", y="spend", color="channel", barmode="group", title="Marketing Spend"), use_container_width=True, key="plot_15")
        st.subheader("CPA (Cost per Order) by Channel")
        st.plotly_chart(px.bar(mk, x="month", y="cpa", color="channel", barmode="group", title="CPA by Channel"), use_container_width=True, key="plot_16")
        st.dataframe(mk[["month","channel","spend","orders","cpa"]].sort_values(["month","channel"]), height=300)
    else:
        st.info("No marketing spend file or not enough data to compute CPA.")

with tab_reviews:
    st.header("Reviews & Rating Analysis")
    def render_review_visuals(users, products, orders, reviews_df, returns_df, fact_df):
        def nonempty(df):
            return (df is not None) and (len(df) > 0)

        if not nonempty(reviews_df):
            st.info("No reviews data available.")
            return

        r = reviews_df.copy()

        # --- Ensure date and rating types ---
        if 'review_date' in r.columns:
            r['review_date'] = pd.to_datetime(r['review_date'], errors='coerce')
            r['review_month'] = r['review_date'].dt.to_period("M").astype(str)
        else:
            r['review_date'] = pd.NaT
            r['review_month'] = np.nan

        # --- Add product_name ---
        if isinstance(products, pd.DataFrame) and {'product_id', 'product_name'}.issubset(products.columns):
            r = r.merge(products[['product_id', 'product_name']].drop_duplicates(), on='product_id', how='left')
        else:
            r['product_name'] = r['product_id'].astype(str)

        # --- Add username (instead of user_id) ---
        if isinstance(users, pd.DataFrame) and {'user_id', 'username'}.issubset(users.columns):
            r = r.merge(users[['user_id', 'username']].drop_duplicates(), on='user_id', how='left')
        else:
            r['username'] = r['user_id'].astype(str)

        # --- Add order status / return info ---
        if isinstance(orders, pd.DataFrame) and 'order_id' in orders.columns:
            if 'status' in orders.columns:
                o2 = orders[['order_id','status']].drop_duplicates()
                r = r.merge(o2, on='order_id', how='left')
                r['is_returned'] = r['status'].isin(["canceled","unavailable"]).astype(int)
            else:
                r['is_returned'] = 0
        else:
            r['is_returned'] = 0

        # --- Review text analysis ---
        if 'review_text' in r.columns:
            r['review_length'] = r['review_text'].astype(str).str.len()
            r['review_words'] = r['review_text'].astype(str).str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
        else:
            r['review_length'] = 0
            r['review_words'] = 0

        # --- Review Stats ---
        st.subheader("Review Statistics")
        avg_rating = r['rating'].mean() if 'rating' in r.columns else np.nan
        total_reviews = r['review_id'].nunique() if 'review_id' in r.columns else len(r)
        reviews_by_month = r.groupby('review_month')['review_id'].nunique().rename('reviews').reset_index() if 'review_month' in r.columns else pd.DataFrame()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Reviews", f"{total_reviews:,}")
        c2.metric("Average Rating", f"{avg_rating:.2f}" if not np.isnan(avg_rating) else "N/A")
        c3.metric("Avg Review Length", f"{r['review_length'].mean():.0f}")

        # --- Product Wise Rating ---
        st.subheader("Product Wise Rating")
        top_prods = (
            r.groupby(['product_id','product_name'])
            .agg(reviews=('review_id','nunique'), avg_rating=('rating','mean'))
            .reset_index()
            .sort_values('reviews', ascending=False)
            .head(20)
        )
        if not top_prods.empty:
            fig_r4 = px.bar(top_prods, x='product_name', y='reviews', hover_data=['avg_rating'], title='Top Products by Review Count')
            beautify_plot(fig_r4)
            st.dataframe(top_prods.style.format({'reviews':'{:,}','avg_rating':'{:.2f}'}), height=250)
        else:
            st.info("No product review data.")

        # --- Reviews per Month ---
        st.subheader("Reviews Per Month")
        if not reviews_by_month.empty:
            fig_r2 = px.line(reviews_by_month, x='review_month', y='reviews', markers=True, title='Reviews per Month')
            beautify_plot(fig_r2)
            st.plotly_chart(fig_r2, use_container_width=True, key="rev_plot_02")
        else:
            st.info("Review dates missing or not parseable.")

        # --- Top Reviewers (using username) ---
        st.subheader("Top Reviewers (by review count)")
        ru = r.groupby(['user_id','username'])['review_id'].nunique().reset_index(name='reviews').sort_values('reviews', ascending=False).head(20)
        if not ru.empty:
            fig_r12 = px.bar(ru, x='username', y='reviews', title='Top Reviewers')
            beautify_plot(fig_r12)
            st.plotly_chart(fig_r12, use_container_width=True, key="rev_plot_12")
        else:
            st.info("No user-level review data.")

        # --- Rating vs Review Length ---
        st.subheader("Rating vs Review Length")
        if {'rating','review_length'}.issubset(r.columns):
            fig_r8 = px.scatter(r, x='review_length', y='rating', hover_data=['product_name','username'], title='Rating vs Review Length')
            beautify_plot(fig_r8)
            st.plotly_chart(fig_r8, use_container_width=True, key="rev_plot_08")
        else:
            st.info("Insufficient columns for rating vs review length.")

        # --- Rating Distribution ---
        st.subheader("Rating Distribution for Top Products")
        tp = top_prods.head(10)['product_id'].tolist()
        if tp:
            sample = r[r['product_id'].isin(tp)]
            fig_r15 = px.box(sample, x='product_name', y='rating', title='Rating Distribution for Top Products')
            beautify_plot(fig_r15)
            st.plotly_chart(fig_r15, use_container_width=True, key="rev_plot_15")
        else:
            st.info("Not enough top products to display distribution.")

        # --- Lowest Rating Record ---
        st.subheader("Lowest Rating Record")
        if {'rating','review_text','username'}.issubset(r.columns):
            r['review_date'] = pd.to_datetime(r['review_date'], errors='coerce')
            r['review_date'] = r['review_date'].dt.strftime("%d %b %Y %I:%M:%S %p")

            low_reviews = r[r['rating'] <= 2].sort_values('review_date', ascending=False).head(20)
            if not low_reviews.empty:
                display_cols = [c for c in ['review_id','review_date','username','product_name','rating','review_text'] if c in low_reviews.columns]
                st.dataframe(low_reviews[display_cols])
            else:
                st.info("No low-rated reviews available to show.")
        else:
            st.info("Review text, rating, or username data not available.")

    render_review_visuals(users, products, orders, reviews, returns, fact)

with st.expander("All Visualization Graphs "):
    def render_25_visuals(users, warehouses, products, orders, order_items, reviews, returns, fact):
        # st.header("All 25 Visuals (Company-wide)")

        def nonempty(df):
            return (df is not None) and (len(df) > 0)

        f = fact.copy() if isinstance(fact, pd.DataFrame) else pd.DataFrame()
        if f.empty:
            st.info("No data to render the visuals.")
            return

        f["order_date"] = pd.to_datetime(f["order_date"], errors="coerce")
        f["month"] = f["order_date"].dt.to_period("M").astype(str)
        orders2 = orders.copy() if isinstance(orders, pd.DataFrame) else pd.DataFrame()
        if "order_date" not in orders2.columns and "order_purchase_timestamp" in orders2.columns:
            orders2["order_date"] = pd.to_datetime(orders2["order_purchase_timestamp"], errors="coerce")
        orders2["month"] = pd.to_datetime(orders2["order_date"], errors="coerce").dt.to_period("M").astype(str)

        st.subheader("01 — Revenue per Order (Histogram)")
        rev_per_order = f.groupby("order_id")["net_line_revenue"].sum().reset_index()
        st.plotly_chart(px.histogram(rev_per_order, x="net_line_revenue", nbins=50, title="Revenue per Order"), use_container_width=True, key="plot_17")

        st.subheader("02 — Orders by Status")
        if "status" in orders2.columns:
            s = orders2["status"].value_counts().reset_index()
            s.columns = ["status","count"]
            st.plotly_chart(px.bar(s, x="status", y="count", title="Orders by Status"), use_container_width=True, key="plot_18")
        else:
            st.info("No status column found in orders.")

        st.subheader("03 — Average Order Value (Monthly)")
        aov_m = f.groupby("month")["net_line_revenue"].sum().reset_index()
        ord_m = f.groupby("month")["order_id"].nunique().reset_index()
        aov_m = aov_m.merge(ord_m, on="month", how="left")
        aov_m["AOV"] = aov_m["net_line_revenue"] / aov_m["order_id"]
        st.plotly_chart(px.line(aov_m, x="month", y="AOV", markers=True, title="AOV Monthly"), use_container_width=True, key="plot_19")

        st.subheader("04 — Revenue by Category (Top 15)")
        cat_rev = f.groupby("category")["net_line_revenue"].sum().sort_values(ascending=False).head(15).reset_index()
        st.plotly_chart(px.bar(cat_rev, x="category", y="net_line_revenue", title="Revenue by Category").update_layout(yaxis_tickprefix="$"), use_container_width=True, key="plot_20")

        st.subheader("05 — Gross Profit by Category")
        cat_gp = f.groupby("category")["gross_profit"].sum().sort_values(ascending=False).reset_index()
        st.plotly_chart(px.bar(cat_gp, x="category", y="gross_profit", title="Gross Profit by Category").update_layout(yaxis_tickprefix="$"), use_container_width=True, key="plot_21")

        st.subheader("06 — Revenue by Warehouse (Seller)")
        if isinstance(warehouses, pd.DataFrame) and {'warehouse_id', 'warehouse_name'}.issubset(warehouses.columns):
            f = f.merge(warehouses[['warehouse_id', 'warehouse_name']].drop_duplicates(), on='warehouse_id', how='left')
        else:
            # fallback if warehouse_name is missing
            f['warehouse_name'] = f['warehouse_id'].astype(str)
        # --- Group and plot ---
        wh_rev = (
            f.groupby(['warehouse_id', 'warehouse_name'], dropna=False)['net_line_revenue']
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig_wh = px.bar(
            wh_rev,
            x="warehouse_name",
            y="net_line_revenue",
            title="Revenue by Warehouse (Seller)",
            text_auto='.2s'
        )
        fig_wh.update_layout(yaxis_tickprefix="$")
        beautify_plot(fig_wh)
        st.plotly_chart(fig_wh, use_container_width=True, key="plot_22")

        st.subheader("07 — Gross Profit by Warehouse")
        if 'warehouse_name' not in f.columns:
            if (
                isinstance(warehouses, pd.DataFrame)
                and {'warehouse_id', 'warehouse_name'}.issubset(warehouses.columns)
            ):
                # Merge warehouse_name from warehouses.csv
                f = f.merge(
                    warehouses[['warehouse_id', 'warehouse_name']].drop_duplicates(),
                    on='warehouse_id',
                    how='left'
                )
            else:
                # Fallback to string ID if name missing
                f['warehouse_name'] = f['warehouse_id'].astype(str)
        wh_gp = (
            f.groupby(['warehouse_id', 'warehouse_name'], dropna=False)['gross_profit']
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig_gp = px.bar(
            wh_gp,
            x="warehouse_name",
            y="gross_profit",
            title="Gross Profit by Warehouse",
            text_auto='.2s'
        )
        fig_gp.update_layout(yaxis_tickprefix="$")
        beautify_plot(fig_gp)
        st.plotly_chart(fig_gp, use_container_width=True, key="plot_23")

        st.subheader("08 — Revenue by Channel")
        if "channel" in f.columns:
            ch = f.groupby("channel")["net_line_revenue"].sum().reset_index()
            st.plotly_chart(px.bar(ch, x="channel", y="net_line_revenue", title="Revenue by Channel").update_layout(yaxis_tickprefix="$"), use_container_width=True, key="plot_24")

        st.subheader("09 — Weekly Revenue Trend")
        weekly = f.groupby(pd.Grouper(key="order_date", freq="W"))["net_line_revenue"].sum().reset_index()
        st.plotly_chart(px.line(weekly, x="order_date", y="net_line_revenue", markers=True, title="Weekly Revenue"), use_container_width=True, key="plot_25")

        st.subheader("10 — Monthly Gross Profit Trend")
        gp_m = f.groupby("month")["gross_profit"].sum().reset_index()
        st.plotly_chart(px.line(gp_m, x="month", y="gross_profit", markers=True, title="Monthly Gross Profit").update_layout(yaxis_tickprefix="$"), use_container_width=True, key="plot_26")

        st.subheader("11 — Top Products by Revenue (Top 15)")
        top_prod_rev = f.groupby(["product_id","sku"]).agg(revenue=("net_line_revenue","sum")).reset_index().sort_values("revenue", ascending=False).head(15)
        st.plotly_chart(px.bar(top_prod_rev, x="sku", y="revenue", title="Top Products by Revenue").update_layout(yaxis_tickprefix="$"), use_container_width=True, key="plot_27")

        st.subheader("12 — Top Products by Margin (Top 15)")
        top_prod_mg = f.groupby(["product_id","sku"]).agg(margin=("gross_profit","sum")).reset_index().sort_values("margin", ascending=False).head(15)
        st.plotly_chart(px.bar(top_prod_mg, x="sku", y="margin", title="Top Products by Margin").update_layout(yaxis_tickprefix="$"), use_container_width=True, key="plot_28")

        st.subheader("13 — Popularity vs Profitability")
        pq = f.groupby(["product_id","sku"]).agg(qty=("qty","sum"), avg_margin=("gross_profit","mean")).reset_index()
        st.plotly_chart(px.scatter(pq, x="qty", y="avg_margin", hover_data=["sku"], title="Qty vs Avg Margin"), use_container_width=True, key="plot_29")

        st.subheader("14 — Price vs Unit Cost (Products)")
        p2 = products.copy() if isinstance(products, pd.DataFrame) and not products.empty else pd.DataFrame()
        if not p2.empty and "unit_price" in p2.columns and "unit_cost" in p2.columns:
            st.plotly_chart(px.scatter(p2, x="unit_price", y="unit_cost", hover_data=["sku","category"], title="Unit Price vs Unit Cost"), use_container_width=True, key="plot_30")
        else:
            st.info("Unit price/cost missing for Price vs Cost chart.")

        st.subheader("15 — Category Revenue Share")
        st.plotly_chart(px.pie(cat_rev, names="category", values="net_line_revenue", title="Category Revenue Share"), use_container_width=True, key="plot_31")

        st.subheader("16 — Return/Cancel Rate by Month")
        o = orders2.copy() if isinstance(orders2, pd.DataFrame) else pd.DataFrame()
        if "status" in o.columns:
            tot = o.groupby("month")["order_id"].nunique().rename("orders_total")
            ret = o[o["status"].isin(["canceled","unavailable"])].groupby("month")["order_id"].nunique().rename("orders_returned")
            rr = pd.concat([tot, ret], axis=1).fillna(0).reset_index()
            rr["return_rate"] = (rr["orders_returned"] / rr["orders_total"]).replace([np.inf, np.nan], 0)
            st.plotly_chart(px.line(rr, x="month", y="return_rate", markers=True, title="Return Rate by Month"), use_container_width=True, key="plot_32")
        else:
            st.info("No status column in orders to compute return rates.")

        st.subheader("17 — Rating Distribution")
        if nonempty(reviews) and "rating" in reviews.columns:
            rd = reviews["rating"].value_counts().sort_index().reset_index()
            rd.columns = ["rating","count"]
            st.plotly_chart(px.bar(rd, x="rating", y="count", title="Rating Distribution"), use_container_width=True, key="plot_33")
        else:
            st.info("No reviews data to show rating distribution.")

        st.subheader("18 — Reviews per Month")
        if nonempty(reviews) and "review_date" in reviews.columns:
            r2 = reviews.copy()
            r2["rm"] = pd.to_datetime(r2["review_date"], errors="coerce").dt.to_period("M").astype(str)
            rpm = r2.groupby("rm")["review_id"].nunique().reset_index(name="reviews")
            st.plotly_chart(px.line(rpm, x="rm", y="reviews", markers=True, title="Reviews per Month"), use_container_width=True, key="plot_34")
        else:
            st.info("No reviews date data available.")

        st.subheader("19 — Return Rate vs Rating")
        if nonempty(reviews) and "status" in orders2.columns:
            orr = orders2[["order_id","status"]].copy()
            orr["is_returned"] = orr["status"].isin(["canceled","unavailable"]).astype(int)
            rj = reviews.merge(orr, on="order_id", how="left")
            byrat = rj.groupby("rating")["is_returned"].mean().reset_index()
            st.plotly_chart(px.line(byrat, x="rating", y="is_returned", markers=True, title="Return Rate vs Rating"), use_container_width=True, key="plot_35")
        else:
            st.info("Cannot compute Return Rate vs Rating (missing reviews or order status).")

        st.subheader("20 — Top Cities by Users")
        if nonempty(users) and "city" in users.columns:
            uc = users["city"].value_counts().head(20).reset_index()
            uc.columns = ["city","users"]
            st.plotly_chart(px.bar(uc, x="city", y="users", title="Top Cities by Users"), use_container_width=True, key="plot_36")
        else:
            st.info("Users data missing or no city column.")

        st.subheader("21 — Orders by Customer State")
        if isinstance(orders, pd.DataFrame) and "customer_state" in orders.columns:
            st_state = orders["customer_state"].value_counts().reset_index()
            st_state.columns = ["state","orders"]
            st.plotly_chart(px.bar(st_state, x="state", y="orders", title="Orders by Customer State"), use_container_width=True, key="plot_37")
        else:
            st.info("No customer_state column in orders.")

        st.subheader("22 — Shipping Cost vs Order Revenue")
        ord_rev = f.groupby("order_id")[["net_line_revenue","ship_share"]].sum().reset_index()
        ord_rev["shipping_cost"] = ord_rev["ship_share"]
        st.plotly_chart(px.scatter(ord_rev, x="net_line_revenue", y="shipping_cost", title="Shipping Cost vs Order Revenue"), use_container_width=True, key="plot_38")

        st.subheader("23 — Discount vs Order Revenue")
        disc_ord = f.groupby("order_id")[["net_line_revenue","discount_share"]].sum().reset_index()
        disc_ord["discount_amount"] = disc_ord["discount_share"]
        st.plotly_chart(px.scatter(disc_ord, x="net_line_revenue", y="discount_amount", title="Discount vs Order Revenue"), use_container_width=True, key="plot_39")

        st.subheader("24 — Revenue vs Gross Profit (Monthly)")
        rev_m = f.groupby("month")["net_line_revenue"].sum().rename("revenue").reset_index()
        mg_m = f.groupby("month")["gross_profit"].sum().rename("gross_profit").reset_index()
        both = rev_m.merge(mg_m, on="month", how="left")
        st.plotly_chart(px.bar(both, x="month", y=["revenue","gross_profit"], barmode="group", title="Revenue & Gross Profit (Monthly)").update_layout(yaxis_tickprefix="$"), use_container_width=True, key="plot_40")

        st.subheader("25 — Category-level Revenue vs Margin")
        cat_sc = f.groupby("category").agg(revenue=("net_line_revenue","sum"), margin=("gross_profit","sum")).reset_index()
        st.plotly_chart(px.scatter(cat_sc, x="revenue", y="margin", hover_data=["category"], title="Category Revenue vs Margin").update_layout(xaxis_tickprefix="$", yaxis_tickprefix="$"), use_container_width=True, key="plot_41")

    render_25_visuals(users, warehouses, products, orders, order_items, reviews, returns, fact)
