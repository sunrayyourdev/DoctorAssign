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
        cursor.execute("DROP TABLE IF EXISTS SQLUser.DoctorApproval CASCADE;")
        
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
                chat_timestamp DATETIME,
                messages VARCHAR(MAX),
                responses VARCHAR(MAX),
                FOREIGN KEY (patientId) REFERENCES SQLUser.Patient(patientId) ON DELETE CASCADE
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

        # DoctorApproval Table
        cursor.execute("""
            CREATE TABLE SQLUser.DoctorApproval (
                doctorId INT NULL,
                chatId INT,
                patientId INT,
                approval BIT DEFAULT 0,
                FOREIGN KEY (doctorId) REFERENCES SQLUser.Doctor(doctorId) ON DELETE CASCADE,
                FOREIGN KEY (chatId) REFERENCES SQLUser.PatientChat(chatId) ON DELETE CASCADE,
                FOREIGN KEY (patientId) REFERENCES SQLUser.Patient(patientId) ON DELETE CASCADE
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

def insert_patientchat_data(patientchat_csv, chatmessage_csv, chatresponse_csv):
    """Reads CSV files and inserts data into the PatientChat table."""
    conn = get_iris_connection()
    if not conn:
        return

    df_patientchat = pd.read_csv(patientchat_csv)
    df_chatmessage = pd.read_csv(chatmessage_csv)
    df_chatresponse = pd.read_csv(chatresponse_csv)

    # Merge chat message and response data into the patient chat DataFrame
    df_patientchat = df_patientchat.merge(df_chatmessage[['chatId', 'content', 'timestamp']], on='chatId', how='left')
    df_patientchat = df_patientchat.merge(df_chatresponse[['chatId', 'content', 'timestamp']], on='chatId', how='left', suffixes=('_message', '_response'))

    # Combine messages and responses into lists
    df_patientchat['messages'] = df_patientchat.apply(lambda row: json.dumps([{"content": row['content_message'], "timestamp": row['timestamp_message']}]), axis=1)
    df_patientchat['responses'] = df_patientchat.apply(lambda row: json.dumps([{"content": row['content_response'], "timestamp": row['timestamp_response']}]), axis=1)

    # Use the first message timestamp as the chat timestamp
    df_patientchat['chat_timestamp'] = df_patientchat['timestamp_message']

    # Drop unnecessary columns
    df_patientchat.drop(columns=['content_message', 'timestamp_message', 'content_response', 'timestamp_response'], inplace=True)

    cursor = conn.cursor()
    placeholders = ", ".join(["?" for _ in df_patientchat.columns])
    query = f"INSERT INTO SQLUser.PatientChat ({', '.join(df_patientchat.columns)}) VALUES ({placeholders})"

    values = [tuple(row) for row in df_patientchat.itertuples(index=False, name=None)]

    try:
        cursor.executemany(query, values)
        conn.commit()
        print(f"✅ Inserted {len(df_patientchat)} rows into SQLUser.PatientChat")
    except Exception as e:
        print(f"❌ Error inserting into SQLUser.PatientChat: {e}")
    finally:
        cursor.close()
        conn.close()

def populate_doctorapproval_table():
    """Populates the DoctorApproval table with data from the PatientChat table."""
    conn = get_iris_connection()
    if not conn:
        return

    cursor = conn.cursor()

    try:
        print("Populating DoctorApproval table...")

        cursor.execute("SELECT chatId, patientId FROM SQLUser.PatientChat")
        patient_chats = cursor.fetchall()

        for chat in patient_chats:
            chat_id, patient_id = chat
            cursor.execute("INSERT INTO SQLUser.DoctorApproval (chatId, patientId) VALUES (?, ?)", (chat_id, patient_id))

        conn.commit()
        print("DoctorApproval table populated successfully.")
    
    except Exception as e:
        print(f"Error populating DoctorApproval table: {e}")

    cursor.close()
    conn.close()

# Directory where cleaned CSVs are stored
CLEANED_DATA_DIR = "cleaned_data"

# Mapping CSV files to correct schema tables
CSV_TABLE_MAPPING = {
    "Doctor_Cleaned.csv": ("SQLUser.Doctor", ["doctorId", "name", "specialty", "DoctorContact", "locationId", "experience_years", "available_hours", "description", "embedding_vector"]),
    "Patient_Cleaned.csv": ("SQLUser.Patient", ["patientId", "email", "name", "age", "gender", "drug_allergies", "medical_conditions"]),
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

    # Step 4: Insert data into the PatientChat table
    patientchat_csv = os.path.join(CLEANED_DATA_DIR, "PatientChat.csv")
    chatmessage_csv = os.path.join(CLEANED_DATA_DIR, "ChatMessage.csv")
    chatresponse_csv = os.path.join(CLEANED_DATA_DIR, "ChatResponse.csv")

    if os.path.exists(patientchat_csv) and os.path.exists(chatmessage_csv) and os.path.exists(chatresponse_csv):
        print(f"Importing data into SQLUser.PatientChat...")
        insert_patientchat_data(patientchat_csv, chatmessage_csv, chatresponse_csv)
    else:
        print(f"Skipping import, one or more files not found.")

    # Step 5: Populate the DoctorApproval table
    populate_doctorapproval_table()