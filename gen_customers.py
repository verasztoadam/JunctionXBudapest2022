import pandas as pd

df_events = pd.read_csv("data/gen/reduced_events.csv")
df_events["customer"] = df_events["customer"].astype(str)
df_checkouts = pd.read_csv("data/gen/checkouts.csv")
df_checkouts["customer"] = df_checkouts["customer"].astype(str)

# Get list of customers
customers = df_events["customer"].unique()

# Create customers table
df_customers = pd.DataFrame(columns=["customer",
                                     "checkout_count",
                                     "coupon_count",
                                     "first_checkout",
                                     "last_checkout",
                                     "spent",
                                     "coupon_spent",
                                     "avg_spent",
                                     "avg_coupon_spent",
                                     "avg_checkout_days"])
for i, customer in enumerate(customers):
    print(f"> Progress index: {i}")
    customer_checkouts = df_checkouts[df_checkouts.customer == customer].sort_values(by="unix_timestamp", ascending=True)
    coupon_checkouts = customer_checkouts[
        (customer_checkouts["properties.coupon_code"].notnull()) |
        (customer_checkouts["properties.gift_card"].notnull() &
         (customer_checkouts["properties.gift_card"] != "[]"))]

    first_checkout = None
    last_checkout = None
    avg_checkout_days = None
    if len(customer_checkouts) > 0:
        first_checkout = int(customer_checkouts.iloc[0]["unix_timestamp"])
        last_checkout = int(customer_checkouts.iloc[-1:]["unix_timestamp"])
        avg_checkout_days = (last_checkout - first_checkout) / len(customer_checkouts) / (60 * 60 * 24)

    spent = customer_checkouts["properties.total"].sum()
    coupon_spent = coupon_checkouts["properties.total"].sum()
    avg_spent = 0
    avg_coupon_spent = 0
    if len(customer_checkouts) > 0:
        avg_spent = spent / len(customer_checkouts)
    if len(coupon_checkouts) > 0:
        avg_coupon_spent = coupon_spent / len(coupon_checkouts)

    df_customers.loc[len(df_customers.index)] = [
        customer,
        len(customer_checkouts),
        len(coupon_checkouts),
        first_checkout,
        last_checkout,
        spent,
        coupon_spent,
        avg_spent,
        avg_coupon_spent,
        avg_checkout_days
    ]

# Save to csv
df_customers.to_csv("./data/gen/customers.csv", sep=',', index=False)
