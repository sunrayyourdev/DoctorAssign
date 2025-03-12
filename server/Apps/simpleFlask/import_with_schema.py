from dotenv import load_dotenv
import iris
import pandas as pd
import os
import json

# IRIS Database Connection Details
IRIS_HOST = "localhost"
IRIS_PORT = 1972
IRIS_NAMESPACE = "USER"
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

# Function to drop all tables
def drop_tables():
    conn = get_iris_connection()
    if not conn:
        return

    cursor = conn.cursor()
    
    try:
        print("Dropping tables...")
        cursor.execute("DROP TABLE IF EXISTS SQLUser.ChatMessage CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS SQLUser.ChatResponse CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS SQLUser.PatientChat CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS SQLUser.Patient CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS SQLUser.Doctor CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS SQLUser.Location CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS SQLUser.Admin CASCADE;")
        
        conn.commit()
        print("All tables dropped successfully.")
    
    except Exception as e:
        print(f"Error dropping tables: {e}")

    cursor.close()
    conn.close()

# Function to create tables with SERIAL (auto-increment) for primary keys
def create_tables():
    conn = get_iris_connection()
    if not conn:
        return

    cursor = conn.cursor()

    try:
        print("Creating tables...")

        # Doctor Table
        cursor.execute("""
            CREATE TABLE SQLUser.Doctor (
                doctorId SERIAL PRIMARY KEY,
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
            CREATE TABLE SQLUser.Patient (
                patientId SERIAL PRIMARY KEY,
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
            CREATE TABLE SQLUser.PatientChat (
                chatId SERIAL PRIMARY KEY,
                patientId INT,
                Title VARCHAR(255),
                FOREIGN KEY (patientId) REFERENCES SQLUser.Patient(patientId) ON DELETE CASCADE
            )
        """)

        # ChatMessage Table
        cursor.execute("""
            CREATE TABLE SQLUser.ChatMessage (
                messageId SERIAL PRIMARY KEY,
                chatId INT,
                content TEXT,
                timestamp DATETIME,
                FOREIGN KEY (chatId) REFERENCES SQLUser.PatientChat(chatId) ON DELETE CASCADE
            )
        """)

        # ChatResponse Table
        cursor.execute("""
            CREATE TABLE SQLUser.ChatResponse (
                responseId SERIAL PRIMARY KEY,
                chatId INT,
                content TEXT,
                timestamp DATETIME,
                FOREIGN KEY (chatId) REFERENCES SQLUser.PatientChat(chatId) ON DELETE CASCADE
            )
        """)

        # Location Table
        cursor.execute("""
            CREATE TABLE SQLUser.Location (
                locationId SERIAL PRIMARY KEY,
                clinicName VARCHAR(255),
                postalCode VARCHAR(20),
                medications TEXT,
                procedures TEXT
            )
        """)

        # Admin Table
        cursor.execute("""
            CREATE TABLE SQLUser.Admin (
                adminId SERIAL PRIMARY KEY,
                name VARCHAR(255),
                admin_role VARCHAR(50)
            )
        """)

        conn.commit()
        print("All tables created successfully.")
    
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

        # ✅ Ensure the embedding vector remains a valid JSON array
        if "embedding_vector" in columns:
            idx = columns.index("embedding_vector")
            if isinstance(row[idx], str):
                row[idx] = row[idx]  # Keep as-is
            else:
                row[idx] = json.dumps(row[idx])  # Convert list to JSON string

        values.append(tuple(row))

    try:
        cursor.executemany(query, values)
        conn.commit()
        print(f"✅ Inserted {len(df)} rows into {table_name}")
    except Exception as e:
        print(f"❌ Error inserting into {table_name}: {e}")
    finally:
        cursor.close()
        conn.close()

# Directory where cleaned CSVs are stored
CLEANED_DATA_DIR = "cleaned_data"

# Mapping CSV files to correct schema tables
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
    drop_tables()  # Step 1: Drop all tables
    create_tables()  # Step 2: Recreate tables

    # Step 3: Insert data
    for csv_file, (table_name, columns) in CSV_TABLE_MAPPING.items():
        csv_path = os.path.join(CLEANED_DATA_DIR, csv_file)
        if os.path.exists(csv_path):
            print(f"Importing {csv_file} into {table_name}...")
            insert_csv_to_iris(csv_path, table_name, columns)
        else:
            print(f"Skipping {csv_file}, file not found.")
