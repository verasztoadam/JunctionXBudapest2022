import pandas as pd

# Read CSV to pandas Dataframe
df = pd.read_csv("./data/gen/reduced_events.csv")

# Select checkout actions with SGD currency
checkouts = df.loc[(df["action"] == 'checkout') & (df["properties.currency"] == "SGD")]

# Print selected Dataframe to .CSV
checkouts.to_csv("./data/gen/checkouts.csv", sep=',', index=False)
