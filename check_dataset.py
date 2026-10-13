import pandas as pd

file_path = "german.data"

df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")

print("\n===== DATASET SHAPE =====")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum().sum())

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

print("\n===== TARGET DISTRIBUTION =====")
print(df.iloc[:, -1].value_counts())

print("\n===== UNIQUE VALUES =====")
for column in df.columns:
    print(f"Column {column}: {df[column].nunique()} unique values")