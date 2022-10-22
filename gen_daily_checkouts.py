import pandas as pd

# Read CSV to pandas Dataframe
df = pd.read_csv("./data/gen/events.csv")

# Select checkout actions with SGD currency
df_checkouts = df.loc[(df["action"] == 'checkout') & (df["properties.currency"] == "SGD")]
df_checkouts = df_checkouts.sort_values(by='date')
dates = df_checkouts["date"].unique()

print(dates)
print(len(dates))


# Create daily checkouts table
df_daily_checkouts = pd.DataFrame(columns=["date", "checkout_count", "coupon_count", "spent", "coupon_spent"])
for i, date in enumerate(dates):
    print(f"> Progress index: {i}")
    daily_checkouts = df_checkouts[df_checkouts.date == date]
    coupon_checkouts = daily_checkouts[
        (daily_checkouts["properties.coupon_code"].notnull()) |
        (daily_checkouts["properties.gift_card"].notnull() &
        (daily_checkouts["properties.gift_card"] != "[]"))]


    df_daily_checkouts.loc[len(df_daily_checkouts.index)] = [
        date,
        len(daily_checkouts),
        len(coupon_checkouts),
        daily_checkouts["properties.total"].sum(),
        coupon_checkouts["properties.total"].sum()
    ]

print(df_daily_checkouts)

# Save to csv
df_daily_checkouts.to_csv("./data/gen/daily_checkouts.csv", sep=',', index=False)