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

nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

@app.route('/')
def homepage():
    return render_template('index.html') 


# ---- DATABASE CONFIG ----
namespace="USER"
port = "1972"
hostname= "localhost"
connection_string = f"{hostname}:{port}/{namespace}"
username = "demo"
password = "demo"

# Load AI embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# OpenAI API Key (Use environment variables in production)
openai.api_key = "your-api-key-here"

# ---- CRUD OPERATIONS ----
@app.route('/create', methods=['POST'])
def create():
    tableName = request.json.get('tableName')
    schema = request.json.get('schema')
    conn = iris.connect(connection_string, username, password)
    cursor = conn.cursor()
    try:
        cursor.execute(f"DROP TABLE {tableName}")
    except:
        pass
    try:
        cursor.execute(f"CREATE TABLE {tableName} {schema}")
    except Exception as inst:
        return jsonify({"response": str(inst)})
    cursor.close()
    conn.commit()
    conn.close()
    return jsonify({"response": "table created"})

@app.route('/getall', methods=['POST'])
def getall():
    tableName = request.json.get('tableName')
    conn = iris.connect(connection_string, username, password)
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT * FROM {tableName}")
        data = cursor.fetchall()
    except Exception as inst:
        return jsonify({"response": str(inst)})
    cursor.close()
    conn.commit()
    conn.close()
    return jsonify({"response": data})

@app.route('/insert', methods=['POST'])
def insert():
    tableName = request.json.get('tableName')
    columns = request.json.get('columns')
    data = request.json.get('data')
    json_compatible_string = data.replace("(", "[").replace(")", "]").replace("'", '"')
    data = json.loads(json_compatible_string)
    qMarks = "(" + ",".join(["?"] * len(data[0])) + ")"
    query = f"INSERT INTO {tableName} {columns} VALUES {qMarks}"
    conn = iris.connect(connection_string, username, password)
    cursor = conn.cursor()
    try:
        cursor.executemany(query, data)
    except Exception as inst:
        return jsonify({"response": str(inst)}) 
    cursor.close()
    conn.commit()
    conn.close()
    return jsonify({"response": "new information added"})

# ---- AI-POWERED DOCTOR RECOMMENDATION ----
@app.route('/recommend_doctor', methods=['POST'])
def recommend_doctor():
    data = request.json
    symptoms = data.get('symptoms', '')

    # Generate embeddings
    symptoms_vector = embedding_model.encode(symptoms).tolist()

    # Fetch doctors from IRIS
    conn = iris.connect(connection_string, username, password)
    cursor = conn.cursor()
    cursor.execute("SELECT doctorId, name, specialty, locationId, experience_years, available_hours, description, embedding_vector FROM SQLUser.Doctor")
    doctors = cursor.fetchall()
    cursor.close()
    conn.close()

    best_match = None
    highest_similarity = -1

    for doctor in doctors:
        doctor_vector = json.loads(doctor[7]) if doctor[7] else [0] * 384  # Default zero vector
        similarity = np.dot(symptoms_vector, doctor_vector) / (np.linalg.norm(symptoms_vector) * np.linalg.norm(doctor_vector))
        
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = doctor

    if best_match:
        return jsonify({
            "doctorId": best_match[0],
            "name": best_match[1],
            "specialty": best_match[2],
            "locationId": best_match[3],
            "experience_years": best_match[4],
            "available_hours": best_match[5],
            "description": best_match[6]
        })
    else:
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
    data = request.json
    patient_input = data.get('message', '')

    cleaned_input = preprocess_text(patient_input)  # Preprocess before sending

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a healthcare chatbot assisting patients."},
                  {"role": "user", "content": cleaned_input}]
    )

    return jsonify({"response": response['choices'][0]['message']['content']})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5010)
