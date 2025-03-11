from dotenv import load_dotenv
import iris
import pandas as pd
import os
import json

# IRIS Database Connection Details
IRIS_HOST = "localhost"
IRIS_PORT = 1972
IRIS_NAMESPACE = "USER"  # Use "SQLUser" if needed
IRIS_USERNAME = "demo"
IRIS_PASSWORD = "demo"

# Function to establish IRIS connection
def get_iris_connection():
    try:
        conn = iris.connect(f"{IRIS_HOST}:{IRIS_PORT}/{IRIS_NAMESPACE}", IRIS_USERNAME, IRIS_PASSWORD)
        return conn
    except Exception as e:
        print(f"Error connecting to IRIS: {e}")
        return None

# Function to manually create tables with PK & FK constraints
def create_tables():
    conn = get_iris_connection()
    if not conn:
        return

    cursor = conn.cursor()

    try:
        # Doctor Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS SQLUser.Doctor (
                doctorId INT PRIMARY KEY,
                name VARCHAR(255),
                specialty VARCHAR(255),
                DoctorContact VARCHAR(50),
                locationId INT,
                experience_years INT,
                available_hours VARCHAR(255),
                description TEXT,
                embedding_vector TEXT
            )
        """)

        # Patient Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS SQLUser.Patient (
                patientId INT PRIMARY KEY,
                email VARCHAR(255),
                name VARCHAR(255),
                age INT,
                gender VARCHAR(10),
                drug_allergies TEXT,
                medical_conditions TEXT
            )
        """)

        # PatientChat Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS SQLUser.PatientChat (
                chatId INT PRIMARY KEY,
                patientId INT,
                Title VARCHAR(255),
                FOREIGN KEY (patientId) REFERENCES SQLUser.Patient(patientId)
            )
        """)

        # ChatMessage Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS SQLUser.ChatMessage (
                messageId INT PRIMARY KEY,
                chatId INT,
                content TEXT,
                timestamp DATETIME,
                FOREIGN KEY (chatId) REFERENCES SQLUser.PatientChat(chatId)
            )
        """)

        # ChatResponse Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS SQLUser.ChatResponse (
                responseId INT PRIMARY KEY,
                chatId INT,
                content TEXT,
                timestamp DATETIME,
                FOREIGN KEY (chatId) REFERENCES SQLUser.PatientChat(chatId)
            )
        """)

        # Location Table (Fixed)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS SQLUser.Location (
                locationId INT PRIMARY KEY,
                clinicName VARCHAR(255),  -- Fixed column name from Address
                postalCode VARCHAR(20),
                medications TEXT,
                procedures TEXT
            )
        """)

        # Admin Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS SQLUser.Admin (
                adminId INT PRIMARY KEY,
                name VARCHAR(255),
                admin_role VARCHAR(50)
            )
        """)

        conn.commit()
        print("All tables created successfully with PK & FK.")
    
    except Exception as e:
        print(f"Error creating tables: {e}")

    cursor.close()
    conn.close()

def insert_csv_to_iris(csv_path, table_name, columns):
    """Reads a CSV and inserts data into the IRIS table with pre-generated embeddings."""
    conn = get_iris_connection()
    if not conn:
        return

    df = pd.read_csv(csv_path)

    # ✅ Drop any accidental extra columns before inserting into IRIS
    df = df.loc[:, df.columns.intersection(columns)]

    # ✅ Ensure correct column order
    df = df[columns]  

    cursor = conn.cursor()
    placeholders = ", ".join(["?" for _ in columns])
    query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"

    values = []
    for row in df.itertuples(index=False, name=None):
        row = list(row)

        # ✅ Ensure the embedding vector remains a valid JSON array (not a double-encoded string)
        if "embedding_vector" in columns:
            idx = columns.index("embedding_vector")
            if isinstance(row[idx], str):  # Check if it's already a string
                row[idx] = row[idx]  # Keep it as-is (assuming CSV is correctly formatted)
            else:
                row[idx] = json.dumps(row[idx])  # Only convert if it's a raw list

        values.append(tuple(row))

    print(f"📝 SQL Query: {query}")
    print(f"📊 First Row Processed: {values[0] if values else 'No Data'}")

    try:
        cursor.executemany(query, values)
        conn.commit()
        print(f"✅ Successfully inserted {len(df)} rows into {table_name}")
    except Exception as e:
        print(f"❌ Error inserting into {table_name}: {e}")
    finally:
        cursor.close()
        conn.close()

# Directory where cleaned CSVs are stored
CLEANED_DATA_DIR = "cleaned_data"

# Mapping CSV files to correct schema tables (Updated for embedding_vector and Location)
CSV_TABLE_MAPPING = {
    "Doctor_Cleaned.csv": ("SQLUser.Doctor", ["doctorId", "name", "specialty", "DoctorContact", "locationId", "experience_years", "available_hours", "description", "embedding_vector"]),
    "Patient_Cleaned.csv": ("SQLUser.Patient", ["patientId", "email", "name", "age", "gender", "drug_allergies", "medical_conditions"]),
    "PatientChat.csv": ("SQLUser.PatientChat", ["chatId", "patientId", "Title"]),
    "ChatMessage.csv": ("SQLUser.ChatMessage", ["messageId", "chatId", "content", "timestamp"]),
    "ChatResponse.csv": ("SQLUser.ChatResponse", ["responseId", "chatId", "content", "timestamp"]),
    "Location.csv": ("SQLUser.Location", ["locationId", "clinicName", "postalCode", "medications", "procedures"]),
    "Admin.csv": ("SQLUser.Admin", ["adminId", "name", "admin_role"]),
}

# Run the import process
if __name__ == "__main__":
    create_tables()  # Step 1: Ensure tables exist

    for csv_file, (table_name, columns) in CSV_TABLE_MAPPING.items():
        csv_path = os.path.join(CLEANED_DATA_DIR, csv_file)
        if os.path.exists(csv_path):
            print(f"Importing {csv_file} into {table_name}...")
            insert_csv_to_iris(csv_path, table_name, columns)
        else:
            print(f"Skipping {csv_file}, file not found.")