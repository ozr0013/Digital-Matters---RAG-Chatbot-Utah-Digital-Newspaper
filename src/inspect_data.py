import pandas as pd

def inspect_csv(file_path, nrows=5, chunksize=100000):
    """
    Inspect a large CSV file without loading the entire dataset.
    
    Args:
        file_path (str): Path to the CSV file
        nrows (int): Number of rows to preview
        chunksize (int): Number of rows per chunk when streaming
    """
    try:
        # Option A: Try reading just first N rows
        df = pd.read_csv(file_path, nrows=nrows)
        print("Preview using nrows")
        print("Columns:", df.columns.tolist())
        print(df.head())
    except Exception as e:
        print("Direct nrows read failed, trying chunks...")
        # Option B: Stream in chunks if nrows fails
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            print("Preview using chunks")
            print("Columns:", chunk.columns.tolist())
            print(chunk.head())
            break

if __name__ == "__main__":
    file_path = "D:/UDN_Project/udn.csv"  # <-- update if needed
    inspect_csv(file_path)
