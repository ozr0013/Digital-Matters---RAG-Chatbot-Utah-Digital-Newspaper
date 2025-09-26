import pandas as pd
from typing import List, Dict

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into overlapping chunks.
    
    Args:
        text (str): The input text to split.
        chunk_size (int): Max number of characters per chunk.
        overlap (int): Overlap between chunks to preserve context.
    
    Returns:
        List[str]: A list of text chunks.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap  # move window forward with overlap

    return chunks


def process_csv(input_csv: str, output_csv: str, text_col: str = "ocr_t", sample: bool = False, sample_size: int = 100) -> None:
    """
    Reads the UDN CSV, chunks the OCR text, and saves a new CSV of chunks.
    
    Args:
        input_csv (str): Path to input CSV (the Solr dump).
        output_csv (str): Path to output CSV with chunks.
        text_col (str): Column name containing text.
        sample (bool): Whether to only process a sample of rows.
        sample_size (int): Number of rows to sample if sample=True.
    """
    print(f"📖 Reading data from {input_csv} ...")
    
    if sample:
        df = pd.read_csv(input_csv, nrows=sample_size, low_memory=False)
        print(f"⚡ Running in SAMPLE mode → using first {sample_size} rows")
    else:
        df = pd.read_csv(input_csv, low_memory=False)
        print(f"⚡ Running in FULL mode → processing {len(df)} rows")

    records = []
    for _, row in df.iterrows():
        text = row.get(text_col, "")
        chunks = chunk_text(str(text))
        for i, chunk in enumerate(chunks):
            records.append({
                "id": row.get("id", None),
                "article_title": row.get("article_title_t", None),
                "date": row.get("date_tdt", None),
                "paper": row.get("paper_t", None),
                "chunk_index": i,
                "text_chunk": chunk
            })

    chunk_df = pd.DataFrame(records)
    print(f"✅ Created {len(chunk_df)} chunks from {len(df)} rows")

    chunk_df.to_csv(output_csv, index=False)
    print(f"💾 Saved to {output_csv}")


if __name__ == "__main__":
    # Example usage (adjust path to your udn.csv on D: drive)
    input_path = r"D:\UDN_Project\udn.csv"
    output_path = r"D:\UDN_Project\udn_chunks.csv"

    # 🔹 Run in SAMPLE mode first for safety
    #process_csv(input_path, output_path, sample=True, sample_size=100)

    # 🔹 Once confirmed, switch to full dataset:
    process_csv(input_path, output_path, sample=False)
