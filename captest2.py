import pandas as pd

csv_file = "Jan-27-2025_Feb-23-2025_541651602373282 (1).csv"  # Replace with actual file path

try:
    df = pd.read_csv(csv_file)
    print("✅ CSV Loaded Successfully!")
    print(df.head())  # Show first 5 rows
    print("Columns:", df.columns)  # Print all columns in CSV
except Exception as e:
    print(f"❌ Error loading CSV: {e}")

# Print function reference
print(pd.read_csv)

print(df.shape)  
print(df.head())  

