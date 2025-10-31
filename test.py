import psycopg2   # pip install psycopg2
import csv

# Database connection parameters
DB_HOST = 'localhost'
DB_NAME = 'test1'
DB_USER = 'postgres'
DB_PASSWORD = 'admin'
DB_PORT = '5432'

# Function to connect to PostgreSQL
def connect_to_db():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )

# Main function to ingest data
def ingest_data():
    # Connect to PostgreSQL
    conn = connect_to_db()
    cur = conn.cursor()

    # Open the CSV file
    with open('data.csv', 'r') as file:
        data_reader = csv.reader(file)
        next(data_reader)  # Skip the header row

        # Insert each row into the table
        for row in data_reader:
            print("Row Data:", row)  # Debugging line

            try:
                # Adjust the column indices based on actual CSV structure
                id_value = int(row[1])  # Assuming the second column is the actual ID
                name_value = row[2]     # Assuming the third column is the Name
                age_value = int(row[3])  # Assuming the fourth column is the Age
                
                cur.execute("INSERT INTO staff (id, name, age) VALUES (%s, %s, %s)", 
                            (id_value, name_value, age_value))

            except (ValueError, IndexError) as e:
                print(f"Skipping row {row} due to error: {e}")

    # Commit and close the connection
    conn.commit()
    cur.close()
    conn.close()
    print("Data ingested successfully")

if __name__ == "__main__":
    ingest_data()
