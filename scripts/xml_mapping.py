
import pandas as pd
from pathlib import Path

def parse_xlsx_mapping(xlsx_path):
    """
    Reads an XLSX file with columns: image_id, well, tile, slice, channel, tiff_file
    Returns: mapping (well -> tile -> slice -> channel -> tiff_file), DataFrame
    """
    df = pd.read_excel(xlsx_path)
    # Ensure columns are strings for mapping keys
    df['well'] = df['well'].astype(str)
    df['tile'] = df['tile'].astype(str)
    df['slice'] = df['slice'].astype(str)
    df['channel'] = df['channel'].astype(str)
    mapping = {}
    for _, row in df.iterrows():
        well = row['well']
        tile = row['tile']
        zslice = row['slice']
        channel = row['channel']
        tiff_file = row['tiff_file'] if 'tiff_file' in row else row[df.columns[5]]
        mapping.setdefault(well, {})
        mapping[well].setdefault(tile, {})
        mapping[well][tile].setdefault(zslice, {})
        mapping[well][tile][zslice][channel] = tiff_file
    return mapping, df
