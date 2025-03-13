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

# Function to drop the PatientChat table
def drop_patientchat_table():
    conn = get_iris_connection()
    if not conn:
        return

    cursor = conn.cursor()
    
    try:
        print("Dropping PatientChat table...")
        cursor.execute("DROP TABLE IF EXISTS SQLUser.PatientChat CASCADE;")
        
        conn.commit()
        print("PatientChat table dropped successfully.")
    
    except Exception as e:
        print(f"Error dropping PatientChat table: {e}")

    cursor.close()
    conn.close()

# Function to create the PatientChat table with combined columns
def create_patientchat_table():
    conn = get_iris_connection()
    if not conn:
        return

    cursor = conn.cursor()

    try:
        print("Creating PatientChat table...")

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

        conn.commit()
        print("PatientChat table created successfully.")
    
    except Exception as e:
        print(f"Error creating PatientChat table: {e}")

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

# Function to create the DoctorApproval table
def create_doctorapproval_table():
    conn = get_iris_connection()
    if not conn:
        return

    cursor = conn.cursor()

    try:
        print("Creating DoctorApproval table...")

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
        print("DoctorApproval table created successfully.")
    
    except Exception as e:
        print(f"Error creating DoctorApproval table: {e}")

    cursor.close()
    conn.close()

def populate_doctorapproval_table():
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

# Run the import process
if __name__ == "__main__":
    # drop_patientchat_table()  # Step 1: Drop the PatientChat table
    # create_patientchat_table()  # Step 2: Recreate the PatientChat table
    create_doctorapproval_table()  # Step 3: Create the DoctorApproval table
    populate_doctorapproval_table()  # Step 4: Populate the DoctorApproval table

    # Step 5: Insert data into the PatientChat table
    # patientchat_csv = os.path.join(CLEANED_DATA_DIR, "PatientChat.csv")
    # chatmessage_csv = os.path.join(CLEANED_DATA_DIR, "ChatMessage.csv")
    # chatresponse_csv = os.path.join(CLEANED_DATA_DIR, "ChatResponse.csv")

    # if os.path.exists(patientchat_csv) and os.path.exists(chatmessage_csv) and os.path.exists(chatresponse_csv):
    #     print(f"Importing data into SQLUser.PatientChat...")
    #     insert_patientchat_data(patientchat_csv, chatmessage_csv, chatresponse_csv)
    # else:
    #     print(f"Skipping import, one or more files not found.")