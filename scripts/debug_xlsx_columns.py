# XLSX Column Debugger for napari_viewer.py
# This script prints the columns of the selected XLSX file to help debug KeyError issues.
import pandas as pd
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
xlsx_path = filedialog.askopenfilename(title='Select XLSX mapping file', filetypes=[('Excel files', '*.xlsx')])
if not xlsx_path:
    print('No XLSX file selected. Exiting.')
    exit(1)
df = pd.read_excel(xlsx_path)
print('XLSX columns:', list(df.columns))
print('First few rows:')
print(df.head())
