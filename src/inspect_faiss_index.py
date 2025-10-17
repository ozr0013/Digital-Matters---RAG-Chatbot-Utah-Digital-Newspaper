import faiss

# Path to your test index
index_path = r"D:\UDN_Project\udn_faiss_test.index"

# Load the index
print("Loading FAISS index...")
index = faiss.read_index(index_path)

# Show basic info
print("Index loaded successfully")
print("Total vectors stored:", index.ntotal)
print("Vector dimension:", index.d)
