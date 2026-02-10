"""
Configuration settings for the Utah Digital Newspapers RAG Chatbot.
Update the paths below to match your environment.
"""

import os

# ======================
# DATA PATHS
# ======================

# Directory containing chunked CSV files (udn_chunks_part*.csv)
CHUNK_DIR = r"E:\UDN_Project\chunked"

# Directory containing embedding files (udn_chunks_part*.npy)
EMB_DIR = r"E:\UDN_Project\embeddings"

# ChromaDB persistent storage location
CHROMA_PATH = r"E:\UDN_Project\chroma_db"

# ======================
# CHROMADB SETTINGS
# ======================

# Collection name in ChromaDB
COLLECTION_NAME = "udn_archive"

# Batch size for database operations
BATCH_SIZE = 2000

# ======================
# MODEL SETTINGS
# ======================

# Sentence transformer model for embeddings
# Must match the model used to create the embeddings!
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Embedding dimension (384 for all-MiniLM-L6-v2)
EMBEDDING_DIM = 384

# ======================
# SEARCH SETTINGS
# ======================

# Default number of results to return
DEFAULT_TOP_K = 5

# Maximum number of results allowed
MAX_TOP_K = 20

# ======================
# FLASK SETTINGS
# ======================

# Flask debug mode (set to False in production)
FLASK_DEBUG = True

# Server host
FLASK_HOST = "0.0.0.0"

# Server port
FLASK_PORT = 5000
