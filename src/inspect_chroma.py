import time, chromadb

path = "D:/UDN_Project/chroma_db"

while True:
    try:
        client = chromadb.PersistentClient(path=path)
        collection = client.get_collection("udn_archive")
        print("📦 Total documents so far:", collection.count())

        peek = collection.peek(3)
        for i, (doc, meta) in enumerate(zip(peek["documents"], peek["metadatas"]), 1):
            print(f"\n[{i}] {meta.get('pdf', 'N/A')}\n{doc[:200]}...")
        break
    except Exception as e:
        print(f"⚠️ Database busy ({e}). Retrying in 15 s...")
        time.sleep(15)
