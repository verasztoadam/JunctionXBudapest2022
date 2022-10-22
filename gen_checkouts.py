import pandas as pd

MYR_TO_SGD = 0.33

# Read CSV to pandas Dataframe
df = pd.read_csv("./data/gen/reduced_events.csv")

# Select checkout actions with SGD currency
checkouts = df.loc[(df["action"] == 'checkout') & (df["properties.currency"].notnull())]
checkouts.loc[checkouts["properties.currency"] == "MYR"]["properties.currency"] = checkouts.loc[checkouts["properties.currency"] == "MYR"]["properties.total"] * MYR_TO_SGD

# Print selected Dataframe to .CSV
checkouts.to_csv("./data/gen/checkouts.csv", sep=',', index=False)
