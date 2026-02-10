"""
Text Chunking Module
Splits large text documents into overlapping chunks for embedding.
"""

import pandas as pd
import os


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """Split text into overlapping chunks."""
    if not isinstance(text, str) or not text.strip() or text.lower() == "nan":
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def process_csv_streaming(input_csv, output_dir, text_col="ocr_t", chunk_size_chars=500, overlap=50, rows_per_file=100000):
    """Stream through huge CSV, chunk text, and save into multiple smaller CSVs."""
    os.makedirs(output_dir, exist_ok=True)

    reader = pd.read_csv(input_csv, chunksize=10000, low_memory=False)
    file_index = 0
    total_chunks = 0
    buffer = []

    for batch_i, df in enumerate(reader):
        print(f"Processing batch {batch_i} ...")

        for _, row in df.iterrows():
            text = str(row.get(text_col, "") or "")
            if not text.strip() or text.lower() == "nan":
                continue

            # chunk the article text
            chunks = chunk_text(text, chunk_size_chars, overlap)
            for j, chunk in enumerate(chunks):
                buffer.append({
                    "id": row.get("id"),
                    "article_title": row.get("article_title_t"),
                    "date": row.get("date_tdt"),
                    "paper": row.get("paper_t"),
                    "chunk_index": j,
                    "chunk_text": chunk
                })
                total_chunks += 1

            # when buffer is large enough, save to file
            if len(buffer) >= rows_per_file:
                out_file = os.path.join(output_dir, f"udn_chunks_part{file_index}.csv")
                pd.DataFrame(buffer).to_csv(out_file, index=False)
                print(f"Wrote {len(buffer)} chunks to {out_file}")
                buffer.clear()
                file_index += 1

    # save leftovers at the end
    if buffer:
        out_file = os.path.join(output_dir, f"udn_chunks_part{file_index}.csv")
        pd.DataFrame(buffer).to_csv(out_file, index=False)
        print(f"Wrote remaining {len(buffer)} chunks to {out_file}")

    print(f"Finished. Total chunks created: {total_chunks}")


if __name__ == "__main__":
    input_path = r"D:\UDN_Project\udn.csv"         # path to your Solr export
    output_dir = r"D:\UDN_Project\chunked"         # output directory for smaller files

    process_csv_streaming(
        input_csv=input_path,
        output_dir=output_dir,
        text_col="ocr_t",
        chunk_size_chars=500,
        overlap=50,
        rows_per_file=100000
    )
