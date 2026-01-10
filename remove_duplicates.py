import pandas as pd

# Load your CSV
df = pd.read_csv("tickers_unpivoted.csv")

# Remove duplicate company rows (keep first)
df_unique = df.drop_duplicates(subset=["company"], keep="first")

# Save to a new cleaned file
output_file = "tickers_unique.csv"
df_unique.to_csv(output_file, index=False)

print(f"Saved cleaned CSV to {output_file}")
print(f"Rows before: {len(df)}")
print(f"Rows after: {len(df_unique)}")
