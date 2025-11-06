# etl_olist_to_dashboard.py
import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "olist_raw")
OUT = os.path.join(BASE, "retail_dashboard", "data")
os.makedirs(OUT, exist_ok=True)

def load_olist():
    customers = pd.read_csv(os.path.join(RAW, "olist_customers_dataset.csv"))
    orders = pd.read_csv(os.path.join(RAW, "olist_orders_dataset.csv"), parse_dates=["order_purchase_timestamp","order_approved_at","order_delivered_carrier_date","order_delivered_customer_date","order_estimated_delivery_date"])
    items = pd.read_csv(os.path.join(RAW, "olist_order_items_dataset.csv"))
    products = pd.read_csv(os.path.join(RAW, "olist_products_dataset.csv"))
    payments = pd.read_csv(os.path.join(RAW, "olist_order_payments_dataset.csv"))
    reviews = pd.read_csv(os.path.join(RAW, "olist_order_reviews_dataset.csv"), parse_dates=["review_creation_date","review_answer_timestamp"])
    sellers = pd.read_csv(os.path.join(RAW, "olist_sellers_dataset.csv"))
    trans = pd.read_csv(os.path.join(RAW, "product_category_name_translation.csv"))
    return customers, orders, items, products, payments, reviews, sellers, trans

def build_users(customers, orders):
    first_purchase = orders.groupby("customer_id")["order_purchase_timestamp"].min().rename("created_at")
    users = customers.merge(first_purchase, on="customer_id", how="left")
    users["user_id"] = users["customer_unique_id"]
    users["created_at"] = pd.to_datetime(users["created_at"]).dt.date
    users["city"] = users["customer_city"]
    users["state"] = users["customer_state"]
    users["channel"] = "marketplace"
    return users[["user_id","created_at","city","state","channel"]].drop_duplicates()

def build_warehouses(sellers):
    wh = sellers.copy()
    wh["warehouse_id"] = wh["seller_id"]
    wh["name"] = wh["seller_id"].str.slice(0,8)
    wh = wh.rename(columns={"seller_city":"city","seller_state":"state"})
    return wh[["warehouse_id","name","city","state"]].drop_duplicates()

def build_products(products, items, trans):
    prods = products.merge(trans, on="product_category_name", how="left")
    prods["category"] = prods["product_category_name_english"].fillna(prods["product_category_name"]).fillna("unknown")
    price_per_prod = items.groupby("product_id")["price"].median().rename("unit_price")
    prods = prods.merge(price_per_prod, on="product_id", how="left")
    prods["unit_cost"] = (prods["unit_price"] * 0.7).round(2)
    prods["sku"] = prods["product_id"].str.slice(0,8)
    return prods[["product_id","sku","category","unit_cost","unit_price"]]

def build_orders(orders, customers, items, payments):
    agg = items.groupby("order_id").agg(
        items_price=("price","sum"),
        freight_value=("freight_value","sum"),
        main_seller=("seller_id","first")
    ).reset_index()
    pay = payments.groupby("order_id")["payment_value"].sum().rename("payment_value").reset_index()
    o = orders.merge(agg, on="order_id", how="left").merge(pay, on="order_id", how="left")
    cust_map = customers[["customer_id","customer_unique_id"]].drop_duplicates()
    o = o.merge(cust_map, on="customer_id", how="left")
    o["order_date"] = o["order_purchase_timestamp"]
    o["warehouse_id"] = o["main_seller"]
    o["channel"] = "online"
    o["status"] = o["order_status"]
    o["shipping_cost"] = o["freight_value"].fillna(0.0)
    gross_basket = (o["items_price"].fillna(0) + o["freight_value"].fillna(0))
    o["discount_amount"] = (gross_basket - o["payment_value"].fillna(gross_basket)).clip(lower=0).round(2)
    orders_out = o[["order_id","customer_unique_id","order_date","warehouse_id","channel","status","shipping_cost","discount_amount"]].copy()
    return orders_out.rename(columns={"customer_unique_id":"user_id"})

def build_order_items(items):
    oi = items.copy()
    oi["qty"] = 1
    oi["unit_price"] = oi["price"]
    return oi[["order_id","product_id","qty","unit_price"]]

def build_reviews_simple(reviews, orders, items):
    r = reviews.rename(columns={
        "review_id":"review_id",
        "order_id":"order_id",
        "review_score":"rating",
        "review_comment_message":"review_text",
        "review_creation_date":"review_date"
    }).copy()
    r["review_date"] = pd.to_datetime(r["review_date"]).dt.date
    first_prod = items.sort_values(["order_id","order_item_id"]).drop_duplicates("order_id")[["order_id","product_id"]]
    r = r.merge(first_prod, on="order_id", how="left")
    return r[["review_id","order_id","product_id","rating","review_text","review_date"]]

def build_returns(orders, payments):
    pay = payments.groupby("order_id")["payment_value"].sum().rename("refund_amount").reset_index()
    ret = orders[orders["order_status"].isin(["canceled","unavailable"])][["order_id","order_purchase_timestamp"]].copy()
    ret = ret.merge(pay, on="order_id", how="left")
    ret["processed_date"] = pd.to_datetime(ret["order_purchase_timestamp"]).dt.date
    ret["return_id"] = ret["order_id"].apply(lambda x: f"RT_{x}")
    ret["product_id"] = np.nan
    ret["qty"] = 1
    ret["reason"] = "canceled/unavailable"
    return ret[["return_id","order_id","product_id","qty","reason","processed_date","refund_amount"]]

def build_marketing_spend(orders):
    df = orders.copy()
    df["month"] = pd.to_datetime(df["order_purchase_timestamp"]).dt.to_period("M").astype(str)
    vol = df.groupby("month")["order_id"].nunique().reset_index(name="orders")
    if vol.empty:
        return pd.DataFrame(columns=["month","channel","spend"])
    vol["spend"] = (vol["orders"] / vol["orders"].max() * 50000).round(0)
    out = []
    for _, row in vol.iterrows():
        out.append({"month": row["month"], "channel": "ads", "spend": row["spend"]})
        out.append({"month": row["month"], "channel": "social", "spend": max(0, row["spend"]*0.4)})
    return pd.DataFrame(out)

def main():
    customers, orders, items, products, payments, reviews, sellers, trans = load_olist()
    users = build_users(customers, orders)
    warehouses = build_warehouses(sellers)
    products_out = build_products(products, items, trans)
    orders_out = build_orders(orders, customers, items, payments)
    order_items_out = build_order_items(items)
    reviews_out = build_reviews_simple(reviews, orders, items)
    returns_out = build_returns(orders, payments)
    marketing_out = build_marketing_spend(orders)

    users.to_csv(os.path.join(OUT, "users.csv"), index=False)
    warehouses.to_csv(os.path.join(OUT, "warehouses.csv"), index=False)
    products_out.to_csv(os.path.join(OUT, "products.csv"), index=False)
    orders_out.to_csv(os.path.join(OUT, "orders.csv"), index=False)
    order_items_out.to_csv(os.path.join(OUT, "order_items.csv"), index=False)
    reviews_out.to_csv(os.path.join(OUT, "reviews.csv"), index=False)
    returns_out.to_csv(os.path.join(OUT, "returns.csv"), index=False)
    marketing_out.to_csv(os.path.join(OUT, "marketing_spend.csv"), index=False)

    print("ETL complete. Files written to:", OUT)

if __name__ == "__main__":
    main()
