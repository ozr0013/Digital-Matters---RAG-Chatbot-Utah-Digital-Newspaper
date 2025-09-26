import requests
import pandas as pd
import json

BASE_URL = "https://api.lib.utah.edu/udn/v1"

def fetch_documents(start=0, limit=100):
    endpoint = f"{BASE_URL}/docs/"
    params = {"start": start, "limit": limit}
    resp = requests.get(endpoint, params=params)
    resp.raise_for_status()
    return resp.json()

if __name__ == "__main__":
    # Fetch first 10 documents
    data = fetch_documents(0, 100)

    # Save raw JSON
    with open("udn_docs_sample.json", "w") as f:
        json.dump(data, f, indent=4)

    # Convert to DataFrame and save CSV
    docs = data.get("docs", [])
    df = pd.DataFrame(docs)
    df.to_csv("udn_docs_sample.csv", index=False)

    print("✅ Data saved: udn_docs_sample.json & udn_docs_sample.csv")
