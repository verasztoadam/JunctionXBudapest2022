import pandas as pd

CUSTOMER_START_INDEX = 300000
CUSTOMER_END_INDEX = 400000

df = pd.read_csv("./data/gen/events.csv")
df['customer'] = df['customer'].astype(str)
df = df.sort_values(by='customer')

customers = df['customer'].unique()[CUSTOMER_START_INDEX:CUSTOMER_END_INDEX]
print(customers)

df_reduced = df[df['customer'].isin(customers)]
df_reduced.to_csv("./data/gen/reduced_events.csv", sep=',', index=False)
