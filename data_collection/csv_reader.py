import pandas as pd
from logger import configure_log

logger = configure_log(__name__)

def read_playlist_from_csv_file(file_path):
    try:
        data = pd.read_csv(file_path)
        return data["playlist_url"].tolist()
    except FileNotFoundError:
        raise Exception(f"CSV file not found at {file_path}")
    except KeyError:
        raise Exception(F"CSV file must have a column named")