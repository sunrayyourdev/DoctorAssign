## DoctorAssign
An AI Chatbot for Doctor Recommendations.
DoctorAssign is an AI chatbot that helps users find a suitable doctor based on their needs and location.

#### Setup Instructions
1. Install Docker Desktop
Download and install Docker Desktop.

Open Docker Desktop and ensure it is running.

2. Set Up InterSystems IRIS Database
Run the following command in the terminal to create an IRIS database container:

```sh
docker run -d --name iris-doctorassign -p 1972:1972 -p 52773:52773 \
    -e IRIS_PASSWORD=demo -e IRIS_USERNAME=demo \
    intersystemsdc/iris-community:latest
```

Once completed, Docker Desktop should show:
- A new container: iris-doctorassign
- A new image: intersystemsdc/iris-community:latest

Open Docker Desktop:
Start the `iris-doctorassign` container.

This must be running whenever you use the back-end, or the database will not update.

Access the database UI at:
http://localhost:52773/csp/sys/exp/%25CSP.UI.Portal.SQL.Home.zen?$NAMESPACE=USER

Username: demo

Password: demo

3. Install Dependencies

Navigate to the backend directories:

```sh
cd simpleFlask
```

Then:
```sh
cd server
```

Open requirements.txt in both simpleFlask and server directories.

Update the file path for intersystems_irispython to match the installation directory inside the server folder.

Install dependencies in both directories:
```sh
pip install -r requirements.txt
```

4. Set Up Environment Variables

Create a .env file in the server directory.

Use .env.dev as a reference.

Ensure it contains:
- OpenAI API Key (Needed for chatbot responses and vector generation).
- IRIS Install Directory (Path to the IRIS installation folder).

5. Prepare Data for the Database

Ensure your simpleFlask directory contains these files:
- Doctor.csv
- Patient.csv

Run the data cleaning script:
```sh
python clean_healthcare_data.py
```

If you are on Linux or Mac, grant the script executable permissions using:
```sh
sudo chmod +x clean_healthcare_data.py
```

This generates additional cleaned CSV files:

Admin.csv, ChatMessages.csv, ChatResponse.csv, Doctor_Cleaned.csv,
Location.csv, Patient_Cleaned.csv, PatientChat.csv

Import data into the IRIS database:

``` py 
python import_with_schema.py
```

To verify:
- Visit http://localhost:52773
- Navigate to the database schema SQLUser
- Confirm that these tables are created:
    - SQLUser.Admin
    - SQLUser.Doctor
    - SQLUser.DoctorApproval
    - SQLUser.Location
    - SQLUser.Patient
    - SQLUser.PatientChat

6. Start the Backend Server

Run the Flask backend:

```py
python flaskBackend.py
```

### API Endpoints and Usage
1. Chatbot Interaction

Post: `http://127.0.0.1:5010/chatbot_response`

Description: Converse with the chatbot.

Example Request:
```json
{
    "patientId": 535,
    "content": "Hello How are you!"
}
```

Example Response:
```json
{
    "chat_history": [
        {
            "content": "Hello How are you!",
            "role": "user"
        },
        {
            "content": "Hello! I'm an AI designed to assist with healthcare. How can I help you today?",
            "role": "assistant"
        }
    ],
    "response": "Hello again. How can I assist you?"
}
```

2. Doctor Recommendation

Post: `http://127.0.0.1:5010/recommend_doctor`

Description: Matches a patient with a doctor based on symptoms.

Example Request:

```json
{
    "patientId": "919"
}
```

Example Response:

```json
{
    "chatgpt_analysis": "The patient might have an allergic reaction...",
    "doctor_details": {
        "doctorId": 270,
        "name": "Dr. Brooks",
        "specialty": "Allergist",
        "experience_years": 21,
        "available_hours": "8 AM - 6 PM",
        "DoctorContact": "drbrooks@yopmail.com"
    }
}
```

3. Doctor Approval for Patient Chat

Post: `http://127.0.0.1:5010/update_doctor_approval/{chatId}`

Description: Updates the database to indicate if the doctor wants to see the patient chat.

Example Request:
```json
{
    "approval": 1
}
```

Example Response:
```json
{
    "response": "Doctor approval status updated successfully"
}
```

4. Testing and Debugging APIs

Get Chat Logs for a Patient:

GET `http://127.0.0.1:5010/get_chat_log/{PatientId}`

Reset Chat Logs for a Patient:

POST `http://127.0.0.1:5010/resetPatientChat/{PatientId}`

#### Final Notes
- Ensure Docker is running before using the backend.
- Use .env for sensitive credentials (DO NOT share it publicly).

For troubleshooting, check Docker logs or Flask error messages.
