import pandas as pd
from datetime import datetime
from dateutil import rrule

# Read CSV to pandas Dataframe
df = pd.read_csv("./data/gen/daily_checkouts.csv")

# Convert pd.date to datetime list
dt_list = []
for date in df['date'].to_list():
    dt_list.append(datetime.strptime(date, '%Y-%m-%dT00:00:00.000Z'))


# Create monthly checkout Dataframe
prev_month = dt_list[0].month
df_monthly_checkouts = pd.DataFrame(columns=["date", "checkout_count", "coupon_count", "spent", "coupon_spent"])

# Init values
monthly_checkouts_cnt = 0
coupon_checkouts_cnt = 0
monthly_spent = 0
coupon_spent = 0

# Iterrate throw rows
for i, row in df.iterrows():
    #print(f"> Progress index: {i}")
    monthly_checkouts_cnt += df['checkout_count'][i]
    coupon_checkouts_cnt += df['coupon_count'][i]
    monthly_spent += df['spent'][i]
    coupon_spent += df['coupon_spent'][i]

    if(dt_list[i].month != prev_month) and i > 1:
        prev_month = dt_list[i].month
        df_monthly_checkouts.loc[len(df_monthly_checkouts.index)] = [
            datetime(year=dt_list[i-1].year, month=dt_list[i-1].month, day=1),
            monthly_checkouts_cnt,
            coupon_checkouts_cnt,
            monthly_spent,
            coupon_spent
        ]
        # Init values
        monthly_checkouts_cnt = 0
        coupon_checkouts_cnt = 0
        monthly_spent = 0
        coupon_spent = 0


print(df_monthly_checkouts)

# Save to csv
df_monthly_checkouts.to_csv("./data/gen/monthly_checkouts.csv", sep=',', index=False)