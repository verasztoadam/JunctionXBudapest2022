import pandas as pd

UTILITY_NAMES = ["unix_timestamp", "date", "customer", "action", "points", "properties.total", "properties.points_burned",
                 "properties.currency", "properties.discount", "properties.shipping", "properties.gift_card",
                 "properties.gift_card_amount", "properties.coupon_code", "used_points", "properties.reward",
                 "properties.level", "properties.purchase_total", "properties.activity", "properties.points"]

# Read CSV to pandas Dataframe
df = pd.read_csv("./data/events_junction.csv")

# Select columns
df = df[UTILITY_NAMES]

# Remove checkout_item rows
df = df[df['action'] != "checkout_item"]

print(df)

# Print selected Dataframe to .CSV
df.to_csv("./data/gen/events.csv", sep=',', index=False)