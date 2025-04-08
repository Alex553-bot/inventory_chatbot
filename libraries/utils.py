import os
import pandas as pd
import csv

def build_path(filename, folder):
    """Builds the full path to a file in the specified folder."""
    return os.path.abspath(os.path.join(folder, filename))

def load_data(file_name, folder='data'):
    """Loads a csv file in a specified folder."""
    file_path = build_path(file_name, folder)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path) 

def save_data(df, file_name, folder='data'):
    """Saves DataFrame to a specified folder."""
    file_path = build_path(file_name, folder)
    df.to_csv(file_path, index=False)
    print(f"Data saved to: {file_path}")

def save_row_csv(data, file_name, folder='data'):
    """Appends rows of data to a CSV file."""
    file_path = build_path(file_name, folder)
    
    if not isinstance(data, list) or not all(isinstance(row, list) for row in data):
        raise ValueError("Data should be a list of lists (rows of data).")
    
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)  # Append the rows to the CSV
    print(f"Rows saved to: {file_path}")  # Print where the rows were saved

def initialize_csv(header, file_name, folder):
    """Initializes a CSV file with a header."""
    file_path = build_path(file_name, folder)
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    print(f"CSV file created with header: {header}")
