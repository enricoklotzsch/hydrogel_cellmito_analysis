
import pandas as pd
from pathlib import Path

def parse_xlsx_mapping(xlsx_path):
    """
    Reads an XLSX file with columns: image_id, well, tile, timepoint, slice, channel, tiff_file
    Returns: mapping (well -> tile -> timepoint -> slice -> channel -> tiff_file), DataFrame
    """
    df = pd.read_excel(xlsx_path)
    if 'timepoint' not in df.columns:
        df['timepoint'] = ''
    # Ensure columns are strings for mapping keys
    df['well'] = df['well'].astype(str)
    df['tile'] = df['tile'].astype(str)
    df['timepoint'] = df['timepoint'].fillna('').astype(str)
    df['slice'] = df['slice'].astype(str)
    df['channel'] = df['channel'].astype(str)
    mapping = {}
    for _, row in df.iterrows():
        well = row['well']
        tile = row['tile']
        timepoint = row['timepoint']
        zslice = row['slice']
        channel = row['channel']
        tiff_file = row['tiff_file'] if 'tiff_file' in row else row[df.columns[5]]
        mapping.setdefault(well, {})
        mapping[well].setdefault(tile, {})
        mapping[well][tile].setdefault(timepoint, {})
        mapping[well][tile][timepoint].setdefault(zslice, {})
        mapping[well][tile][timepoint][zslice][channel] = tiff_file
    return mapping, df
