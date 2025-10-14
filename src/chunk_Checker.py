import pandas as pd
import glob

# Path to your chunked folder
path = r"D:\UDN_Project\chunked\*.csv"

total_chunks = 0
files = glob.glob(path)

for f in files:
    try:
        df = pd.read_csv(f)
        total_chunks += len(df)
        print(f"{f} → {len(df)} chunks")
    except Exception as e:
        print(f"⚠️ Skipped {f}: {e}")

print(f"\n✅ Total actual chunks across all files: {total_chunks:,}")
