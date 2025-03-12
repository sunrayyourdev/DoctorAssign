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

client = openai.OpenAI(api_key="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")

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
        print(" Connected to IRIS database")
        return conn
    except Exception as e:
        print(f" IRIS Connection Error: {e}")
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
    print(f" FAISS Index Initialized with {faiss_index.d} dimensions")  # Should print 1536

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
            print(f" Doctor ID: {doctor[0]}, Embedding Length: {len(embedding_vector)}")  # Should print 1536

            doc_text = f"Doctor: {doctor[1]}\nSpecialty: {doctor[2]}\nExperience: {doctor[4]} years\nDescription: {doctor[6]}"

            doc = Document(text=doc_text, metadata={  #  REMOVE embeddings from metadata
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
            print(f" Warning: Invalid embedding vector format for Doctor ID {doctor[0]} - {e}")

    # Insert all embeddings into FAISS
    if embeddings:
        faiss_index.add(np.vstack(embeddings))  #  Efficient batch insertion
        print(f" Successfully inserted {len(embeddings)} embeddings into FAISS.")

    # Build the index with doctor data
    if documents:
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, settings=Settings)
        print(" Doctor Index Built Successfully")
        return index
    else:
        print(" Warning: No doctors found in the database.")
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
    """Finds the best matching doctor based on patient's latest chat message."""
    data = request.json
    patient_id = data.get('patientId')

    if not patient_id:
        return jsonify({"error": "Missing patientId"}), 400

    try:
        conn = get_iris_connection()
        cursor = conn.cursor()

        # Fetch latest chat messages for the patient
        cursor.execute("""
            SELECT content FROM (
                SELECT CM.content, ROW_NUMBER() OVER (ORDER BY CM.timestamp DESC) AS row_num
                FROM SQLUser.ChatMessage CM
                JOIN SQLUser.PatientChat PC ON CM.chatId = PC.chatId
                WHERE PC.patientId = ?
            ) AS subquery
            WHERE row_num <= 5
        """, (patient_id,))
        chat_data = cursor.fetchall()

        if not chat_data:
            return jsonify({"error": "No recent chat messages found for patient"}), 404

        # Combine messages into one input string
        patient_input = " ".join([row[0] for row in chat_data])
        print(f"Extracted Patient Input: {patient_input}")

        # Generate an embedding for the extracted text
        embedding_vector = embedding_fn.get_text_embedding(patient_input)

        # Ensure embedding vector is correct dimension
        if len(embedding_vector) != 1536:
            return jsonify({"error": "Embedding dimension mismatch"}), 500

        embedding_np = np.array([embedding_vector], dtype=np.float32)

        # Check if FAISS index has stored doctor embeddings
        if faiss_index.ntotal == 0:
            return jsonify({"error": "No doctor embeddings found in FAISS index"}), 500

        # Search FAISS for the closest doctor match
        D, I = faiss_index.search(embedding_np, k=1)

        # Debugging: Log FAISS search results
        print(f"FAISS Search Results: D={D}, I={I}")

        if I[0][0] == -1:
            return jsonify({"error": "No matching doctor found"}), 404

        # Ensure FAISS result is valid
        if I[0][0] >= len(index.docstore.docs):
            return jsonify({"error": "FAISS returned an invalid index"}), 500

        # Get doctor ID mapping
        try:
            doctor_id_mapping = {idx: doc.metadata["doctorId"] for idx, doc in enumerate(index.docstore.docs.values())}
            doctor_id = doctor_id_mapping.get(I[0][0], None)

            if not doctor_id:
                return jsonify({"error": "Doctor ID not found"}), 404
        except Exception as mapping_error:
            return jsonify({"error": f"Doctor ID mapping error: {str(mapping_error)}"}), 500

        # Fetch doctor details from database
        cursor.execute("""
            SELECT doctorId, name, specialty, experience_years, available_hours, description, doctorContact
            FROM SQLUser.Doctor WHERE doctorId = ?
        """, (doctor_id,))
        doctor = cursor.fetchone()

        cursor.close()
        conn.close()

        if doctor:
            return jsonify({
                "doctorId": doctor[0],
                "name": doctor[1],
                "specialty": doctor[2],
                "experience": doctor[3],
                "available_hours": doctor[4],
                "description": doctor[5],
                "doctorContact": doctor[6]
            })
        else:
            return jsonify({"error": "Doctor not found in database"}), 404

    except Exception as e:
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
    """Processes chatbot responses and ensures symptom extraction for doctor recommendation."""
    try:
        start_time = time.time()  # Track execution time

        # Log request data
        data = request.json
        logging.debug(f"Received request: {data}")

        if not data:
            logging.error("Request JSON is empty or missing")
            return jsonify({"error": "Invalid request. No data received."}), 400

        patient_id = data.get('patientId')
        patient_input = data.get('content', '')

        if not patient_id or not patient_input:
            logging.error("Missing required fields: patientId or content")
            return jsonify({"error": "Invalid request. Missing patientId or content."}), 400

        # Connect to IRIS
        conn = get_iris_connection()
        if not conn:
            logging.error("Database connection failed.")
            return jsonify({"error": "Database connection failed"}), 500
        
        cursor = conn.cursor()

        #  Check if patient has an existing chat session
        cursor.execute("SELECT chatId FROM SQLUser.PatientChat WHERE patientId = ?", (patient_id,))
        chat_row = cursor.fetchone()

        if not chat_row:
            logging.info(f"No existing chat session found for patientId {patient_id}. Creating new chat session.")

            # Create a new chat session
            cursor.execute("INSERT INTO SQLUser.PatientChat (patientId, Title) VALUES (?, ?)", (patient_id, "New Chat"))
            conn.commit()

            # Retrieve the newly created chatId
            cursor.execute("SELECT LAST_IDENTITY() FROM SQLUser.PatientChat")
            chat_id_row = cursor.fetchone()
            chat_id = chat_id_row[0] if chat_id_row else None

            if not chat_id:
                logging.error("Failed to retrieve chatId after inserting new chat.")
                return jsonify({"error": "Failed to generate chatId"}), 500

        else:
            chat_id = chat_row[0]
            logging.debug(f"Existing chat session found. chatId: {chat_id}")

        #  Store patient input in chat history
        try:
            cursor.execute("INSERT INTO SQLUser.ChatMessage (chatId, content, timestamp) VALUES (?, ?, NOW())",
                           (chat_id, patient_input))
            conn.commit()
            logging.info(f"Stored patient input for chatId {chat_id}")
        except Exception as e:
            logging.error(f"Error storing patient message: {str(e)}", exc_info=True)
            return jsonify({"error": "Failed to store patient message"}), 500

        #  Fetch latest chat history (IRIS-Compatible Query)
        try:
            cursor.execute("""
                SELECT TOP 5 content FROM SQLUser.ChatMessage 
                WHERE chatId = ? 
                ORDER BY timestamp DESC
            """, (chat_id,))
            
            past_messages = cursor.fetchall()
            if not past_messages:
                logging.warning(f"No past messages found for chatId {chat_id}")
                past_messages = []

            # Convert result to list
            past_messages = [row[0] for row in past_messages]
            logging.debug(f"Past Messages Retrieved: {past_messages}")

        except Exception as e:
            logging.error(f"Error fetching chat history: {str(e)}", exc_info=True)
            return jsonify({"error": "Failed to retrieve chat history"}), 500

        #  Add context and send request to OpenAI
        chat_history = [{"role": "user", "content": msg} for msg in reversed(past_messages)]
        chat_history.append({"role": "user", "content": patient_input})

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a healthcare chatbot. Always ask about symptoms before recommending a doctor."}
                ] + chat_history,
                timeout=10
            )
            chatbot_reply = response.choices[0].message.content  # Corrected new OpenAI response format
            logging.info("Received chatbot response from OpenAI.")
        except Exception as openai_error:
            logging.error(f"Error calling OpenAI API: {openai_error}", exc_info=True)
            chatbot_reply = "Sorry, I couldn't process your request at the moment."

        #  Store chatbot response
        try:
            cursor.execute("INSERT INTO SQLUser.ChatResponse (chatId, content, timestamp) VALUES (?, ?, NOW())",
                           (chat_id, chatbot_reply))
            conn.commit()
            logging.info("Chatbot response stored in database.")

        except Exception as e:
            logging.error(f"Error storing chatbot response: {str(e)}", exc_info=True)
            return jsonify({"error": "Failed to store chatbot response"}), 500

        #  Ensure connections are closed properly
        cursor.close()
        conn.close()

        #  Track and log total execution time
        total_time = time.time() - start_time
        logging.info(f"Request processed successfully in {total_time:.2f} seconds.")

        return jsonify({"response": chatbot_reply})

    except Exception as e:
        logging.error(f"Unexpected error in chatbot_response: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred. Please check the logs for details."}), 500
    except Exception as e:
        logging.error(f"Unexpected error in chatbot_response: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred. Please check the logs for details."}), 500


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)  # Enable DEBUG logs
    app.run(debug=True, host='0.0.0.0', port=5010, use_reloader=False)
