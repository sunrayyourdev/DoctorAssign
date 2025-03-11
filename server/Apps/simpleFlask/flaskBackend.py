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
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings
import time
from collections import OrderedDict
import faiss # Faiss for vector similarity search


# Simple in-memory cache to store chatbot responses
chatbot_cache = OrderedDict()
CACHE_SIZE = 100  # Max number of cached responses
CACHE_TTL = 300  # Cache expiry time in seconds

def get_cached_response(patient_id, message):
    """Retrieve cached response if available and not expired."""
    cache_key = f"{patient_id}:{message}"
    if cache_key in chatbot_cache:
        timestamp, response = chatbot_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:  # Check if cache is still valid
            return response
    return None

def cache_response(patient_id, message, response):
    """Store response in cache."""
    cache_key = f"{patient_id}:{message}"
    chatbot_cache[cache_key] = (time.time(), response)
    if len(chatbot_cache) > CACHE_SIZE:
        chatbot_cache.popitem(last=False)  # Remove oldest entry if cache limit is reached

# Download stopwords for NLP preprocessing 
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

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

# ---- OPENAI CONFIG ----
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Missing required environment variable: OPENAI_API_KEY")

# ---- Free Llama embedding model ----
#embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ---- LlamaIndex+OpenAI Vector Search Setup ----
embedding_fn = OpenAIEmbedding(model_name="text-embedding-ada-002")

# Create a FAISS index for 1536-dimensional vectors
faiss_index = faiss.IndexFlatL2(1536)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# Instead of ServiceContext, create a Settings object
Settings.embed_model = embedding_fn
Settings.chunk_size = 31306  # Increase the chunk size to 4096 or higher
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

    # Debug: Confirm FAISS index initialization
    print(f"✅ FAISS Index Initialized with {faiss_index.d} dimensions")  # Should print 1536

    documents = []
    embeddings = []  # Separate list for FAISS embeddings
    metadata_list = []  # Store metadata separately

    for doctor in doctors:
        try:
            # Convert embedding_vector from string to list
            if doctor[7] and isinstance(doctor[7], str):  
                cleaned_embedding = doctor[7].strip()
                if cleaned_embedding:
                    embedding_vector = json.loads(cleaned_embedding)  # Parse JSON properly
                    if not isinstance(embedding_vector, list):  # Ensure it is a list
                        raise ValueError("Invalid embedding format")
                    embedding_vector = list(map(float, embedding_vector))  # Convert to float
                else:
                    embedding_vector = [0] * 1536  # Default vector
            else:
                embedding_vector = [0] * 1536  # Default vector if missing

            # Debug: Check retrieved embedding length
            print(f"🔍 Doctor ID: {doctor[0]}, Embedding Length: {len(embedding_vector)}")  # Should print 1536

            doc_text = f"Doctor: {doctor[1]}\nSpecialty: {doctor[2]}\nExperience: {doctor[4]} years\nDescription: {doctor[6]}"

            doc = Document(text=doc_text, metadata={  # ✅ REMOVE embeddings from metadata
                "doctorId": doctor[0],
                "name": doctor[1],
                "specialty": doctor[2],
                "locationId": doctor[3],
                "experience_years": doctor[4],
                "available_hours": doctor[5],
                "description": doctor[6],
            })
            documents.append(doc)

            # Store embedding separately for FAISS
            np_embedding = np.array([embedding_vector], dtype=np.float32)  # Convert to NumPy array
            embeddings.append(np_embedding)
            metadata_list.append({"doctorId": doctor[0], "name": doctor[1], "specialty": doctor[2]})

        except (ValueError, json.JSONDecodeError) as e:
            print(f"⚠️ Warning: Invalid embedding vector format for Doctor ID {doctor[0]} - {e}")

    # Insert all embeddings into FAISS
    if embeddings:
        faiss_index.add(np.vstack(embeddings))  # ✅ Efficient batch insertion
        print(f"✅ Successfully inserted {len(embeddings)} embeddings into FAISS.")

    # Build the index with doctor data
    if documents:
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, settings=Settings)
        print("✅ Doctor Index Built Successfully")
        return index
    else:
        print("⚠️ Warning: No doctors found in the database.")
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
                admin_role VARCHAR(50)
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
    data = request.json
    table_name = data.get("tableName")
    columns = data.get("columns")
    values = data.get("values")

    if not table_name or not columns or not values:
        return jsonify({"error": "Missing required fields (tableName, columns, values)"}), 400

    try:
        # Convert list of columns into a SQL string
        column_names = ", ".join(columns)

        # Ensure each value set is formatted correctly
        placeholders = ", ".join(["?" for _ in columns])

        # Final SQL query
        query = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"

        conn = get_iris_connection()
        if not conn:
            return jsonify({"error": "Failed to connect to IRIS"}), 500

        cursor = conn.cursor()
        cursor.executemany(query, values)  # Insert multiple values if provided
        conn.commit()

        cursor.close()
        conn.close()

        return jsonify({"response": "Data inserted successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- AI-POWERED DOCTOR RECOMMENDATION ----
@app.route('/recommend_doctor', methods=['POST'])
def recommend_doctor():
    """Finds the best matching doctor based on symptoms using FAISS vector search."""
    data = request.json
    symptoms = data.get('symptoms', '')

    if not symptoms:
        return jsonify({"error": "Missing symptoms"}), 400

    try:
        # ✅ Generate embedding
        embedding_vector = embedding_fn.get_text_embedding(symptoms)
        embedding_np = np.array([embedding_vector], dtype=np.float32)  # Convert to NumPy array

        # ✅ Search FAISS index for the closest match
        D, I = faiss_index.search(embedding_np, k=1)  # Get the closest doctor match

        # 🛠 Debugging: Print FAISS search results
        print(f"FAISS Index Result: {I[0][0]}")

        if I[0][0] == -1:  # If FAISS returns -1, no match was found
            print("❌ No matching doctor found in FAISS index.")
            return jsonify({"error": "No matching doctor found"}), 404

        # ✅ Ensure we correctly map FAISS indices to doctor IDs
        doctor_id_mapping = {idx: doc.metadata["doctorId"] for idx, doc in enumerate(index.docstore.docs.values())}

        # 🛠 Debugging: Print mapping dictionary
        print(f"Doctor ID Mapping: {doctor_id_mapping}")

        doctor_id = doctor_id_mapping.get(I[0][0])

        if doctor_id is None:
            print(f"❌ No doctor found for FAISS index {I[0][0]}")
            return jsonify({"error": "Doctor ID not found"}), 404

        print(f"✅ Matched Doctor ID: {doctor_id}")

        # ✅ Retrieve doctor details from the database
        conn = get_iris_connection()
        if not conn:
            print("❌ Failed to connect to IRIS database.")
            return jsonify({"error": "Failed to connect to IRIS database"}), 500

        cursor = conn.cursor()

        # 🔹 **IRIS SQL Query Handling**
        try:
            # Ensure correct data type for `doctorId`
            doctor_id = int(doctor_id)  # If doctorId is stored as INTEGER

            cursor.execute("""
                SELECT doctorId, name, specialty, experience_years, available_hours, description
                FROM SQLUser.Doctor
                WHERE doctorId = ?;
            """, (doctor_id,))

            doctor = cursor.fetchone()

        except Exception as db_error:
            print(f"❌ IRIS Query Failed: {db_error}")
            return jsonify({"error": f"Database query failed: {str(db_error)}"}), 500

        finally:
            cursor.close()
            conn.close()

        # ✅ Ensure doctor data is properly formatted
        if doctor:
            try:
                # Convert IRIS `DataRow` object into a Python list
                doctor = list(doctor)
                print(f"✅ Doctor Found: {doctor}")  # Debugging: Check actual values

                return jsonify({
                    "doctorId": int(doctor[0]),
                    "name": str(doctor[1]),
                    "specialty": str(doctor[2]),
                    "experience": int(doctor[3]),
                    "available_hours": str(doctor[4]),
                    "description": str(doctor[5]) if doctor[5] else "No description available"
                })

            except Exception as parse_error:
                print(f"❌ Data Parsing Error: {parse_error}")
                return jsonify({"error": f"Data processing failed: {str(parse_error)}"}), 500

        else:
            print("❌ Doctor not found in database.")
            return jsonify({"error": "Doctor not found in database"}), 404

    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")  # Log unexpected errors
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# ---- AI CHATBOT RESPONSE (WITH PREPROCESSING) ----
def preprocess_text(text):
    text = text.lower().strip()  # Lowercase and trim spaces
    text = re.sub(r'\d+', '', text)  # Remove numbers (optional)
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation

    words = word_tokenize(text)  # Tokenize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]  # Lemmatization
    
    return " ".join(words)


@app.route('/chatbot_response', methods=['POST'])
def chatbot_response():
    """
    Processes chatbot responses by:
      - Fetching past chat history from IRIS for context.
      - Checking a local cache to reduce repeated OpenAI API calls.
      - Storing the conversation in IRIS.
    """
    data = request.json
    patient_id = data.get('patientId')
    patient_input = data.get('message', '')

    if not patient_id or not patient_input:
        return jsonify({"response": "Invalid request"}), 400

    # Check if response is already cached
    cached = get_cached_response(patient_id, patient_input)
    if cached:
        return jsonify({"response": cached})

    # Preprocess the input before sending (your existing preprocess_text function)
    cleaned_input = preprocess_text(patient_input)

    # Retrieve the last 5 patient messages from IRIS to add context
    conn = get_iris_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT content FROM SQLUser.ChatMessage 
        WHERE chatId = (SELECT chatId FROM SQLUser.PatientChat WHERE patientId = ?)
        ORDER BY timestamp DESC LIMIT 5
    """, (patient_id,))
    # Get past messages (most recent first) and then reverse them to preserve chronological order
    past_messages = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    
    chat_history = [{"role": "user", "content": msg} for msg in reversed(past_messages)]
    chat_history.append({"role": "user", "content": cleaned_input})
    
    # Call OpenAI API with system prompt and chat history
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a healthcare chatbot assisting patients."}] + chat_history
    )
    chatbot_reply = response['choices'][0]['message']['content']
    
    # Cache the new response for future use
    cache_response(patient_id, patient_input, chatbot_reply)
    
    # Store the new patient message and chatbot response in IRIS
    conn = get_iris_connection()    
    cursor = conn.cursor()
    cursor.execute("INSERT INTO SQLUser.ChatMessage (chatId, content, timestamp) VALUES (?, ?, NOW())",
                   (patient_id, cleaned_input))
    cursor.execute("INSERT INTO SQLUser.ChatResponse (chatId, content, timestamp) VALUES (?, ?, NOW())",
                   (patient_id, chatbot_reply))
    conn.commit()
    cursor.close()
    conn.close()
    
    return jsonify({"response": chatbot_reply})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5010)