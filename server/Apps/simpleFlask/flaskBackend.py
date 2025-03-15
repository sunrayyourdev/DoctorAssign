from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import iris
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from openai import OpenAI
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
# @app.route('/create_tables', methods=['POST'])
# def create_tables():
#     """Creates all required tables in the InterSystems IRIS database for DoctorAssign using CSV-based schema."""
#     conn = get_iris_connection()
#     cursor = conn.cursor()

#     try:
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS SQLUser.Doctor (
#                 doctorId INT PRIMARY KEY,
#                 name VARCHAR(255),
#                 specialty VARCHAR(255),
#                 doctorContact VARCHAR(50),
#                 locationId INT,
#                 experience_years INT,
#                 available_hours VARCHAR(255),
#                 description TEXT,
#                 embedding_vector TEXT
#             )
#         """)

#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS SQLUser.Patient (
#                 patientId INT PRIMARY KEY,
#                 email VARCHAR(255),
#                 name VARCHAR(255),
#                 age INT,
#                 gender VARCHAR(10),
#                 drug_allergies TEXT,
#                 medical_conditions TEXT
#             )
#         """)

#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS SQLUser.PatientChat (
#                 chatId INT PRIMARY KEY,
#                 patientId INT,
#                 Title VARCHAR(255),
#                 FOREIGN KEY (patientId) REFERENCES SQLUser.Patient(patientId)
#             )
#         """)

#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS SQLUser.ChatMessage (
#                 messageId INT PRIMARY KEY,
#                 chatId INT,
#                 content TEXT,
#                 timestamp DATETIME,
#                 FOREIGN KEY (chatId) REFERENCES SQLUser.PatientChat(chatId)
#             )
#         """)

#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS SQLUser.ChatResponse (
#                 responseId INT PRIMARY KEY,
#                 chatId INT,
#                 content TEXT,
#                 timestamp DATETIME,
#                 FOREIGN KEY (chatId) REFERENCES SQLUser.PatientChat(chatId)
#             )
#         """)

#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS SQLUser.Location (
#                 locationId INT PRIMARY KEY,
#                 clinicName VARCHAR(255),
#                 Address VARCHAR(255),
#                 postalCode VARCHAR(20),
#                 medications TEXT,
#                 procedures TEXT
#             )
#         """)

#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS SQLUser.Admin (
#                 adminId INT PRIMARY KEY,
#                 name VARCHAR(255),
#                 admin_role VARCHAR(50)
#             )
#         """)

#     except Exception as e:
#         return jsonify({"response": f"Error creating tables: {str(e)}"})

#     cursor.close()
#     conn.commit()
#     conn.close()

#     return jsonify({"response": "All tables created successfully"})


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

# ---- AI CHATBOT RESPONSE (WITH PREPROCESSING) ----
def preprocess_text(text):
    text = text.lower().strip()  # Lowercase and trim spaces
    text = re.sub(r'\d+', '', text)  # Remove numbers (optional)
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation

    words = word_tokenize(text)  # Tokenize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]  # Lemmatization
    
    return " ".join(words)


# ---- AI-POWERED DOCTOR RECOMMENDATION ----
@app.route('/recommend_doctor', methods=['POST'])
def recommend_doctor():
    """Finds the best matching doctor and validates it with OpenAI before responding."""
    data = request.json
    patient_id = data.get('patientId')

    if not patient_id:
        return jsonify({"error": "Missing patientId"}), 400

    try:
        conn = get_iris_connection()
        cursor = conn.cursor()

        # Fetch latest chat messages for the patient
        cursor.execute("""
            SELECT chatId, messages FROM SQLUser.PatientChat WHERE patientId = ?
        """, (patient_id,))
        chat_row = cursor.fetchone()

        if not chat_row:
            return jsonify({"error": "No recent chat messages found for patient"}), 404

        chat_id = chat_row[0]
        # Combine messages into one input string
        messages = json.loads(chat_row[1]) if chat_row[1] else []
        patient_input = " ".join([msg["content"] for msg in messages])
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

            # Debugging: Log doctor ID mapping
            print(f"Doctor ID Mapping: {doctor_id_mapping}")
            print(f"Selected Doctor ID: {doctor_id}")

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
        print(f"Recommended Doctor Details: {doctor}")

        if not doctor:
            return jsonify({"error": "Doctor not found in database"}), 404

        # Get column names from the cursor and convert DataRow to dictionary
        columns = [desc[0] for desc in cursor.description]
        doctor_dict = dict(zip(columns, doctor))

        # --- OpenAI ChatGPT Validation ---
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Prompt to analyze patient symptoms and validate recommendation
        prompt = f"""
        Patient's reported symptoms (please extract and consider only the medically relevant content, and ignore any extraneous or non-medical text such as testing messages, repeated statements, or unrelated chatter):
        "{patient_input}"

        Recommended Doctor:
        - Name: {doctor_dict['name']}
        - Specialty: {doctor_dict['specialty']}
        - Experience: {doctor_dict['experience_years']} years
        - Available Hours: {doctor_dict['available_hours']}
        - Description: {doctor_dict['description']}

        Assume that this recommended doctor has been selected by a sophisticated vector search algorithm as the best match. Based solely on the medically relevant symptoms provided above, please answer the following questions:

        1. Is there enough medically relevant information in the patient's reported symptoms to confidently justify that this doctor is a good match? Answer ONLY "Yes" or "No". (Assume that even one clearly stated significant symptom is sufficient.)
        2. If you answer "Yes", please provide a structured explanation that includes:
           - A list of possible conditions that could be inferred from the patient's symptoms.
           - An explanation of why the doctor's specialty is appropriate for these conditions.
           - A brief, patient-friendly summary that explains the recommendation.
        3. If you answer "No", please explain which additional medically relevant details are needed to confidently justify the recommendation.
        """

        # Make OpenAI API call
        openai_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical AI assistant. Provide responsible and accurate medical insights."},
                {"role": "user", "content": prompt}
            ]
        )

        # Extract GPT response
        gpt_reply = openai_response.choices[0].message.content
        print(f"OpenAI Response: {gpt_reply}")

        # Parse the GPT response
        lines = gpt_reply.split("\n")
        first_answer = lines[0].strip()  # First line should be "Yes" or "No"
        cleaned_answer = re.sub(r'^\d+\.\s*', '', first_answer.strip().lower())

        # Instead of rejecting the result if GPT says "No", always trust the vector search.
        # If GPT returns "No", we add a note explaining that more information may be needed.
        additional_message = ""
        if cleaned_answer == "no":
            additional_message = ("We matched you with this doctor; however, we believe that more information may be needed "
                                  "before a proper recommendation can be made. Please consider consulting the doctor for further advice.")

        # Remove leading numbering from subsequent explanation lines
        cleaned_explanation = [
            re.sub(r"^\d+\.\s*", "", line).strip() for line in lines[1:] if line.strip()
        ]

        # Format the explanation with an optional additional message
        formatted_explanation = f"""
{additional_message}

Possible Conditions:
{cleaned_explanation[0] if len(cleaned_explanation) > 0 else "N/A"}

Why this doctor is a good fit:
{cleaned_explanation[1] if len(cleaned_explanation) > 1 else "N/A"}

Patient-friendly recommendation:
{cleaned_explanation[2] if len(cleaned_explanation) > 2 else "N/A"}
"""

        # Combine ChatGPT's explanation with doctor details
        response_data = {
            "doctor_details": doctor_dict,
            "chatgpt_evaluation": cleaned_answer,
            "chatgpt_analysis": formatted_explanation
        }

        # Update the DoctorApproval table
        cursor.execute("""
            UPDATE SQLUser.DoctorApproval
            SET doctorId = ?, approval = 1
            WHERE chatId = ?
        """, (doctor_id, chat_id))
        conn.commit()

        cursor.close()
        conn.close()

        return jsonify(response_data)  # Return enriched recommendation

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

    
@app.route('/chatbot_response', methods=['POST'])
def chatbot_response():
    """
    Processes chatbot responses while maintaining full conversation history in two lists:
      - messages (user messages)
      - responses (assistant responses)

    Uses a sliding window to provide GPT-4 with recent context. Also integrates
    doctor approval initialization if needed, without requiring 'role' in the database.
    """
    try:
        start_time = time.time()
        data = request.json

        if not data:
            return jsonify({"error": "Invalid request. No data received."}), 400

        patient_id = data.get('patientId')
        patient_input = data.get('content', '')

        if not patient_id or not patient_input:
            return jsonify({"error": "Invalid request. Missing patientId or content."}), 400

        # Connect to the IRIS database
        conn = get_iris_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        cursor = conn.cursor()

        # Retrieve existing chat history if it exists
        cursor.execute("SELECT chatId, messages, responses FROM SQLUser.PatientChat WHERE patientId = ?", (patient_id,))
        chat_row = cursor.fetchone()

        if not chat_row:
            # Create a new chat record if none exists
            cursor.execute(
                "INSERT INTO SQLUser.PatientChat (patientId, Title, chat_timestamp, messages, responses) VALUES (?, ?, NOW(), ?, ?)",
                (patient_id, "New Chat", json.dumps([]), json.dumps([]))
            )
            conn.commit()
            cursor.execute("SELECT LAST_IDENTITY() FROM SQLUser.PatientChat")
            chat_id_row = cursor.fetchone()
            chat_id = chat_id_row[0] if chat_id_row else None
            messages = []
            responses = []

            # Also create a new record in DoctorApproval table
            cursor.execute(
                "INSERT INTO SQLUser.DoctorApproval (chatId, patientId, approval) VALUES (?, ?, 0)",
                (chat_id, patient_id)
            )
            conn.commit()
        else:
            chat_id = chat_row[0]
            messages = json.loads(chat_row[1]) if chat_row[1] else []
            responses = json.loads(chat_row[2]) if chat_row[2] else []

        # 1) Append the new user message to messages
        new_user_message = {
            "content": patient_input,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        messages.append(new_user_message)

        # 2) Build a combined conversation by weaving user messages & assistant responses
        def build_conversation(user_msgs, assistant_msgs):
            """
            Merge user and assistant entries into a single conversation list with 'role' keys,
            so GPT-4 knows which text is from the user vs. from the assistant.
            """
            conversation = []
            # Loop up to the longer of the two lists
            for i in range(max(len(user_msgs), len(assistant_msgs))):
                if i < len(user_msgs):
                    conversation.append({
                        "role": "user",
                        "content": user_msgs[i]["content"]
                    })
                if i < len(assistant_msgs):
                    conversation.append({
                        "role": "assistant",
                        "content": assistant_msgs[i]["content"]
                    })
            return conversation
        
        full_conversation = build_conversation(messages, responses)

        # 3) Use a sliding window (e.g., last 5 exchanges) for GPT context
        window_size = 5 * 2  # 5 user messages + 5 assistant replies => 10 total
        limited_conversation = full_conversation[-window_size:]

        # 4) Call GPT-4 with the system prompt and limited conversation
        gpt_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a healthcare chatbot. Maintain conversation history while assisting the patient."}
            ] + limited_conversation,
            timeout=10
        )
        chatbot_reply = gpt_response.choices[0].message.content

        # 5) Append the new assistant message to responses
        new_assistant_message = {
            "content": chatbot_reply,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        responses.append(new_assistant_message)

        # 6) (Optional) Trim the size of responses to match the sliding window, if desired
        # This is not strictly necessary; it just keeps the DB smaller.
        # responses = responses[-5:]  # Keep last 5 assistant replies
        # messages = messages[-5:]    # Keep last 5 user messages

        # 7) Update the chat record in the database
        cursor.execute(
            "UPDATE SQLUser.PatientChat SET messages = ?, responses = ?, chat_timestamp = NOW() WHERE chatId = ?",
            (json.dumps(messages), json.dumps(responses), chat_id)
        )
        conn.commit()
        cursor.close()
        conn.close()

        # 8) Return the chatbot reply + the limited conversation for client display
        return jsonify({"response": chatbot_reply, "chat_history": limited_conversation})
    
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


# For reseting the tested patient chat
@app.route('/resetPatientChat/<int:patient_id>', methods=['POST'])
def resetPatientChat(patient_id):
    """Reset the patient chat log by deleting all messages and responses."""
    conn = get_iris_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cursor = conn.cursor()
    cursor.execute("SELECT chatId FROM SQLUser.PatientChat WHERE patientId = ?", (patient_id,))
    chat_row = cursor.fetchone()

    if not chat_row:
        return jsonify({"error": "No chat log found for the specified patient ID"}), 404

    chat_id = chat_row[0]

    cursor.execute("UPDATE SQLUser.PatientChat SET messages = ?, responses = ? WHERE chatId = ?", (json.dumps([]), json.dumps([]), chat_id))
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"response": "Patient chat log has been reset successfully"})

@app.route('/get_chat_log/<int:patient_id>', methods=['GET'])
def get_chat_log(patient_id):
    """Fetches the chat log for a specific patient."""
    try:
        conn = get_iris_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM SQLUser.PatientChat WHERE patientId = ?", (patient_id,))
        chat_row = cursor.fetchone()

        if not chat_row:
            return jsonify({"error": "No chat log found for the specified patient ID"}), 404

        # Get column names
        columns = [desc[0] for desc in cursor.description]

        # Convert the row to a dictionary
        chat_log = dict(zip(columns, chat_row))

        # Parse messages and responses if they exist
        chat_log['messages'] = json.loads(chat_log['messages']) if chat_log['messages'] else []
        chat_log['responses'] = json.loads(chat_log['responses']) if chat_log['responses'] else []

        cursor.close()
        conn.close()

        return jsonify({"chat_log": chat_log})

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/update_doctor_approval/<int:chat_id>', methods=['POST'])
def update_doctor_approval(chat_id):
    """Updates the approval status of a doctor for a specific patient and chat."""
    data = request.json
    approval = data.get('approval')

    if not chat_id or approval is None:
        return jsonify({"error": "Missing required fields (chatId and approval)"}), 400

    try:
        conn = get_iris_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500

        cursor = conn.cursor()

        # Update the approval status in the DoctorApproval table
        cursor.execute("""
            UPDATE SQLUser.DoctorApproval
            SET approval = ?
            WHERE chatId = ?
        """, (approval, chat_id))
        conn.commit()

        cursor.close()
        conn.close()

        return jsonify({"response": "Doctor approval status updated successfully"})

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
    
# --- Helper Function to Extract Symptoms ---
def extract_symptoms(text):
    """
    Extracts potential symptoms from the provided text using a keyword-based approach.
    Can be expanded with more advanced NLP techniques.
    """
    common_symptoms = [
    "fever", "chills", "sweating", "cough", "shortness of breath", "wheezing",
    "sore throat", "runny nose", "nasal congestion", "headache", "migraine",
    "dizziness", "lightheadedness", "nausea", "vomiting", "diarrhea", "constipation",
    "abdominal pain", "stomach cramps", "back pain", "muscle pain", "joint pain",
    "fatigue", "weakness", "swelling", "edema", "rash", "itching", "hives",
    "redness", "blurred vision", "double vision", "light sensitivity", "loss of taste",
    "loss of smell", "palpitations", "chest pain", "chest tightness", "anxiety",
    "insomnia", "confusion", "memory loss", "tingling", "numbness", "seizures",
    "weight loss", "weight gain", "frequent urination", "painful urination",
    "urinary urgency", "heartburn", "acid reflux", "malaise"
    ]
    found = []
    lower_text = text.lower()
    for symptom in common_symptoms:
        if symptom in lower_text:
            found.append(symptom)
    return list(set(found))  # Ensure unique symptoms

@app.route('/get_patient_chat/<int:doctor_id>/<int:patient_id>', methods=['GET'])
def get_patient_chat(doctor_id, patient_id):
    """
    Securely retrieves a patient's chat history, including messages and chatbot responses.
    Also extracts potential symptoms from patient messages.
    """
    try:
        conn = get_iris_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500

        cursor = conn.cursor()

        #  Security Check: Ensure the doctor is assigned to this patient and approved
        cursor.execute("""
            SELECT doctorId FROM SQLUser.DoctorApproval WHERE patientId = ? AND doctorId = ? AND approval = 1
        """, (patient_id, doctor_id))
        assignment = cursor.fetchone()

        if not assignment:
            return jsonify({"error": "Access denied. You are not assigned to this patient or approval is pending."}), 403

        # Fetch chat logs
        cursor.execute("""
            SELECT chatId, messages, responses FROM SQLUser.PatientChat WHERE patientId = ?
        """, (patient_id,))
        chat_row = cursor.fetchone()

        if not chat_row:
            return jsonify({"error": "No chat log found for this patient"}), 404

        chat_id = chat_row[0]
        messages = json.loads(chat_row[1]) if chat_row[1] else []
        responses = json.loads(chat_row[2]) if chat_row[2] else []

        # Extract Symptoms from Patient Messages
        patient_text = " ".join([msg["content"] for msg in messages])
        symptoms = extract_symptoms(patient_text)

        cursor.close()
        conn.close()

        return jsonify({
            "chatId": chat_id,
            "messages": messages,
            "responses": responses,
            "extracted_symptoms": symptoms  # Added symptom analysis
        })

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
    
@app.route('/get_all_patient_chats/<int:doctor_id>', methods=['GET'])
def get_all_patient_chats(doctor_id):
    """
    Retrieves all patient chat logs that the doctor is approved to view.
    """
    try:
        conn = get_iris_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500

        cursor = conn.cursor()

        # Get all chat IDs where the doctor is approved to view
        cursor.execute("""
            SELECT chatId FROM SQLUser.DoctorApproval
            WHERE doctorId = ? AND approval = 1
        """, (doctor_id,))
        chat_ids = cursor.fetchall()

        if not chat_ids:
            return jsonify({"error": "No approved chats found for this doctor"}), 404

        # Flatten chat_ids to a list of integers
        chat_ids = [chat[0] for chat in chat_ids]

        # Retrieve chat logs for the approved chat IDs
        placeholders = ','.join(['?'] * len(chat_ids))
        query = f"SELECT * FROM SQLUser.PatientChat WHERE chatId IN ({placeholders})"
        cursor.execute(query, chat_ids)
        chat_rows = cursor.fetchall()

        if not chat_rows:
            return jsonify({"error": "No chat logs found"}), 404

        # Map each chat row to a dictionary
        columns = [desc[0] for desc in cursor.description]
        chat_logs = []
        for row in chat_rows:
            chat_log = dict(zip(columns, row))
            chat_log['messages'] = json.loads(chat_log.get('messages', '[]'))
            chat_log['responses'] = json.loads(chat_log.get('responses', '[]'))
            chat_logs.append(chat_log)

        cursor.close()
        conn.close()

        return jsonify({"chat_logs": chat_logs})
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)  # Enable DEBUG logs
    app.run(debug=True, host='0.0.0.0', port=5010, use_reloader=False)
