from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import iris
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import re
import nltk
from nltk.corpus import stopwords
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, ServiceContext, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.vector_stores.simple import SimpleVectorStore

# Download stopwords for NLP preprocessing 
nltk.download('stopwords')

# Initialize Flask App
app = Flask(__name__)
CORS(app)

@app.route('/')
def homepage():
    return render_template('index.html') 

# ---- DATABASE CONNECTION CONFIG ----
load_dotenv()
namespace = os.getenv("IRIS_NAMESPACE", "USER")
hostname = os.getenv("IRIS_HOST", "localhost")
port = os.getenv("IRIS_PORT", "1972")
username = os.getenv("IRIS_USERNAME", "demo")  
password = os.getenv("IRIS_PASSWORD", "demo")

# Function to establish a connection with IRIS
def get_iris_connection():
    """Establishes a secure connection to the IRIS database."""
    try:
        conn = iris.connect(f"{hostname}:{port}/{namespace}", username, password)
        return conn
    except Exception as e:
        print(f"IRIS Connection Error: {e}")
        return None

# ---- AI EMBEDDING MODEL CONFIG ----
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ---- OPENAI CONFIG ----
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Missing required environment variable: OPENAI_API_KEY")

# ---- LlamaIndex Vector Search Setup ----
embedding_fn = OpenAIEmbedding(model_name="text-embedding-ada-002")  # Uses OpenAI for embeddings

# Vector Store Setup
vector_store = SimpleVectorStore()
service_context = ServiceContext.from_defaults(embed_model=embedding_fn)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ---- Build Doctor Index on Startup ----
def build_doctor_index():
    """Fetch all doctors from IRIS and build the LlamaIndex vector search index."""
    conn = get_iris_connection()
    if not conn:
        print("Error: Failed to connect to IRIS database.")
        return None

    cursor = conn.cursor()
    cursor.execute("SELECT doctorId, name, specialty, locationId, experience_years, available_hours, description, embedding_vector FROM SQLUser.Doctor")
    doctors = cursor.fetchall()
    cursor.close()
    conn.close()

    # Prepare doctor data for vector search
    documents = []
    for doctor in doctors:
        doc_text = f"Doctor: {doctor[1]}\nSpecialty: {doctor[2]}\nExperience: {doctor[4]} years\nDescription: {doctor[6]}"

        # Convert embedding_vector from string (JSON) to list
        try:
            embedding_vector = json.loads(doctor[7]) if doctor[7] else [0] * 384
        except json.JSONDecodeError:
            print(f"Warning: Invalid embedding vector format for Doctor ID {doctor[0]}")
            embedding_vector = [0] * 384

        # Create document with structured metadata
        doc = Document(text=doc_text, metadata={
            "doctorId": doctor[0],
            "name": doctor[1],
            "specialty": doctor[2],
            "locationId": doctor[3],
            "experience_years": doctor[4],
            "available_hours": doctor[5],
            "description": doctor[6],
            "embedding": embedding_vector  # Attach precomputed embeddings
        })
        documents.append(doc)

    # Build the index with doctor data
    if documents:
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context)
        print("Doctor Index Built Successfully")
        return index
    else:
        print("Warning: No doctors found in the database.")
        return None

# Initialize index on startup
index = build_doctor_index()


# ---- CRUD OPERATIONS ----
@app.route('/create_tables', methods=['POST'])
def create_tables():
    """Creates all required tables in the InterSystems IRIS database for DoctorAssign using CSV-based schema."""
    conn = get_iris_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS SQLUser.Doctor (
                doctorId INT PRIMARY KEY,
                name VARCHAR(255),
                specialty VARCHAR(255),
                doctorContact VARCHAR(50),
                locationId INT,
                experience_years INT,
                available_hours VARCHAR(255),
                description TEXT,
                embedding_vector TEXT
            )
        """)

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

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS SQLUser.PatientChat (
                chatId INT PRIMARY KEY,
                patientId INT,
                Title VARCHAR(255),
                FOREIGN KEY (patientId) REFERENCES SQLUser.Patient(patientId)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS SQLUser.ChatMessage (
                messageId INT PRIMARY KEY,
                chatId INT,
                content TEXT,
                timestamp DATETIME,
                FOREIGN KEY (chatId) REFERENCES SQLUser.PatientChat(chatId)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS SQLUser.ChatResponse (
                responseId INT PRIMARY KEY,
                chatId INT,
                content TEXT,
                timestamp DATETIME,
                FOREIGN KEY (chatId) REFERENCES SQLUser.PatientChat(chatId)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS SQLUser.Location (
                locationId INT PRIMARY KEY,
                clinicName VARCHAR(255),
                Address VARCHAR(255),
                postalCode VARCHAR(20),
                medications TEXT,
                procedures TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS SQLUser.Admin (
                adminId INT PRIMARY KEY,
                name VARCHAR(255),
                role VARCHAR(50)
            )
        """)

    except Exception as e:
        return jsonify({"response": f"Error creating tables: {str(e)}"})

    cursor.close()
    conn.commit()
    conn.close()

    return jsonify({"response": "All tables created successfully"})


@app.route('/getall', methods=['POST'])
def getall():
    """Fetches all records from a specified table."""
    table_name = request.json.get('tableName')

    if not table_name:
        return jsonify({"response": "Missing table name"}), 400

    conn = get_iris_connection()
    if not conn:
        return jsonify({"response": "Failed to connect to IRIS"}), 500

    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT * FROM {table_name}")
        data = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]  # Get column names
        records = [dict(zip(columns, row)) for row in data]  # Convert to JSON format
    except Exception as e:
        return jsonify({"response": str(e)})
    finally:
        cursor.close()
        conn.commit()
        conn.close()

    return jsonify({"response": records})


@app.route('/insert', methods=['POST'])
def insert():
    """Inserts data into a specified table in IRIS."""
    table_name = request.json.get('tableName')
    columns = request.json.get('columns')
    values = request.json.get('values')  # Expecting a list of tuples

    if not table_name or not columns or not values:
        return jsonify({"response": "Missing required fields (tableName, columns, values)"}), 400

    placeholders = ", ".join(["?"] * len(values[0]))
    query = f"INSERT INTO {table_name} {columns} VALUES ({placeholders})"

    conn = get_iris_connection()
    if not conn:
        return jsonify({"response": "Failed to connect to IRIS"}), 500

    cursor = conn.cursor()
    try:
        cursor.executemany(query, values)
    except Exception as e:
        return jsonify({"response": str(e)})
    finally:
        cursor.close()
        conn.commit()
        conn.close()

    return jsonify({"response": "Data inserted successfully"})

# ---- AI-POWERED DOCTOR RECOMMENDATION ----
@app.route('/recommend_doctor', methods=['POST'])
def recommend_doctor():
    """Finds the best matching doctor based on patient symptoms using LlamaIndex."""
    data = request.json
    symptoms = data.get('symptoms', '')

    # Ensure index is built
    if index is None:
        return jsonify({"response": "Doctor index is not built yet."}), 500

    # Query the index
    query_engine = index.as_query_engine()
    response = query_engine.query(symptoms)

    # Extract structured doctor information
    best_match = response.response
    for doctor in doctors:
        if best_match in doctor[1]:  # Match doctor name
            return jsonify({
                "doctorId": doctor[0],
                "name": doctor[1],
                "specialty": doctor[2],
                "locationId": doctor[3],
                "experience_years": doctor[4],
                "available_hours": doctor[5],
                "description": doctor[6]
            })

    return jsonify({"message": "No suitable doctor found"})

# ---- AI CHATBOT RESPONSE (WITH PREPROCESSING) ----
def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'[^\w\s]', '', text)  # Removing punctuation
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]  # Removing stopwords
    return " ".join(words)

@app.route('/chatbot_response', methods=['POST'])
def chatbot_response():
    """Processes chatbot responses and stores conversation in IRIS."""
    data = request.json
    patient_id = data.get('patientId', None)
    patient_input = data.get('message', '')

    cleaned_input = preprocess_text(patient_input)  # Preprocess before sending

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a healthcare chatbot assisting patients."},
                  {"role": "user", "content": cleaned_input}]
    )

    chatbot_reply = response['choices'][0]['message']['content']

    # Store messages in IRIS
    conn = get_iris_connection()
    cursor = conn.cursor()

    # Insert patient message
    cursor.execute("INSERT INTO SQLUser.ChatMessage (chatId, content, timestamp) VALUES (?, ?, NOW())",
                   (patient_id, cleaned_input))

    # Insert chatbot response
    cursor.execute("INSERT INTO SQLUser.ChatResponse (chatId, content, timestamp) VALUES (?, ?, NOW())",
                   (patient_id, chatbot_reply))

    cursor.close()
    conn.commit()
    conn.close()

    return jsonify({"response": chatbot_reply})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5010)
