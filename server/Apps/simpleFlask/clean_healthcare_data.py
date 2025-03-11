from dotenv import load_dotenv
import pandas as pd
import os
import random
from datetime import datetime, timedelta, timezone
import openai
import json
import time

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define input and output paths
base_path = "./"
input_files = {
    "Doctor": os.path.join(base_path, "Doctor.csv"),
    "Patient": os.path.join(base_path, "Patient.csv")
}
output_folder = "cleaned_data"
os.makedirs(output_folder, exist_ok=True)

# Sample list of 20 real clinics in Singapore
clinics = [
    {"clinicName": "CHANG CLINIC & SURGERY", "Address": "7 Everton Park #01-21", "postalCode": "080007"},
    {"clinicName": "CHANGI CLINIC", "Address": "848 Sims Avenue #01-734 Eunosville", "postalCode": "400848"},
    {"clinicName": "SUNBEAM MEDICAL CLINIC", "Address": "443C Fajar Road #01-76", "postalCode": "673443"},
    {"clinicName": "TANG FAMILY CLINIC", "Address": "84 Jalan Jurong Kechil", "postalCode": "598593"},
    {"clinicName": "FRONTIER PEOPLE'S CLINIC PTE LTD", "Address": "Blk 123 Bedok North Street 2 #01-152", "postalCode": "460123"},
    {"clinicName": "Central 24-HR Clinic (Bedok)", "Address": "219 Bedok Central, #01-124", "postalCode": "460219"},
    {"clinicName": "Central 24-HR Clinic (Clementi)", "Address": "450 Clementi Ave 3 #01-291", "postalCode": "120450"},
    {"clinicName": "Central 24-HR Clinic (Hougang)", "Address": "681 Hougang Ave 8, #01-829", "postalCode": "530681"},
    {"clinicName": "Central 24-HR Clinic (Jurong)", "Address": "492 Jurong West Street 41, #01-54", "postalCode": "640492"},
    {"clinicName": "Central 24-HR Clinic (Yishun)", "Address": "701A Yishun Ave 5, #01-04", "postalCode": "761701"},
    {"clinicName": "OneCare Clinic Yishun", "Address": "846 Yishun Ring Rd #01-3669", "postalCode": "760846"},
    {"clinicName": "Acumed Medical Group", "Address": "Blk 727 Ang Mo Kio Ave 6", "postalCode": "560727"},
    {"clinicName": "Bukit Batok Medical Clinic", "Address": "632 Bukit Batok Central", "postalCode": "650632"},
    {"clinicName": "East Coast Family Clinic", "Address": "121 Bedok Reservoir Rd #01-202", "postalCode": "470121"},
    {"clinicName": "Everhealth Clinic & Surgery", "Address": "201D Tampines Street 21 #01-1171", "postalCode": "524201"},
    {"clinicName": "Healthway Medical", "Address": "446 Clementi Ave 3 #01-201", "postalCode": "120446"},
    {"clinicName": "Silver Cross Family Clinic", "Address": "2 Lorong Lew Lian #01-50", "postalCode": "531002"},
    {"clinicName": "The Clinic Group", "Address": "1 Raffles Quay #09-02", "postalCode": "048583"},
    {"clinicName": "Trucare Medical", "Address": "Blk 682 Hougang Ave 4 #01-320", "postalCode": "530682"},
    {"clinicName": "Zion Medical Centre", "Address": "Blk 502 Jurong West Ave 1 #01-815", "postalCode": "640502"}
]

# Lists of medications and procedures
medications_list = ["Paracetamol", "Ibuprofen", "Amoxicillin", "Metformin", "Amlodipine", "Omeprazole", "Atorvastatin", "Salbutamol", "Lisinopril", "Simvastatin"]
procedures_list = ["Blood Test", "ECG", "X-Ray", "Vaccination", "Minor Surgery", "Health Screening", "Physiotherapy", "Endoscopy", "Ultrasound", "Wound Dressing"]

def generate_batch_embeddings(text_list, batch_size=50):
    """Generate embeddings for a batch of texts using OpenAI API with batch handling."""
    all_embeddings = []

    # Ensure OpenAI API key is correctly set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key missing.")
        return [json.dumps([0.0] * 1536)] * len(text_list)  # Return default vectors

    openai.api_key = api_key

    for i in range(0, len(text_list), batch_size):
        batch = text_list[i: i + batch_size]
        print(f"🔄 Requesting embeddings for batch {i // batch_size + 1}/{(len(text_list) // batch_size) + 1}...")

        try:
            response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=batch
            )

            # Ensure embeddings are properly stored as JSON strings
            embeddings = [json.dumps(data.embedding) for data in response.data] if response.data else \
                         [json.dumps([0.0] * 1536)] * len(batch)

            # Ensure correct length
            if len(embeddings) != len(batch):
                print(f"⚠️ Mismatch in embeddings count: Expected {len(batch)}, got {len(embeddings)}")
                embeddings = [json.dumps([0.0] * 1536)] * len(batch)

            all_embeddings.extend(embeddings)

        except Exception as e:
            print(f"⚠️ Error generating embeddings: {e}")
            all_embeddings.extend([json.dumps([0.0] * 1536)] * len(batch))  # Default empty vector

        time.sleep(1)  # Delay to prevent API rate limits

    # Ensure the length of the output matches the input
    if len(all_embeddings) != len(text_list):
        print(f"⚠️ Final length mismatch: Expected {len(text_list)}, got {len(all_embeddings)}")
        while len(all_embeddings) < len(text_list):
            all_embeddings.append(json.dumps([0.0] * 1536))

    return all_embeddings

# Function to assign random medications and procedures
def assign_services(services_list):
    num_services = random.randint(1, 5)  # Each clinic offers between 1 to 5 services
    return "; ".join(random.sample(services_list, num_services))

# Function to generate random working hours for doctors
def generate_available_hours():
    """Randomly generate 10-hour clinic working hours within common operating hours."""
    opening_times = ["7 AM", "8 AM", "9 AM", "10 AM", "11 AM"]
    start_time = random.choice(opening_times)
    
    # Create closing time by adding 10 hours
    closing_hour = int(start_time.split()[0]) + 10
    closing_period = "PM" if closing_hour >= 12 else "AM"
    if closing_hour > 12:
        closing_hour -= 12
    
    closing_time = f"{closing_hour} {closing_period}"
    return f"{start_time} - {closing_time}"

# Function to clean Doctor data
def clean_doctor_data(df):
    """Format Doctor data to match the schema and pre-generate embeddings."""
    df.rename(columns={"DoctorID": "doctorId", "DoctorName": "name", "Specialization": "specialty"}, inplace=True)

    df["locationId"] = range(1, len(df) + 1)  # Assign location IDs
    df["experience_years"] = [random.randint(1, 30) for _ in range(len(df))]  # Random experience
    df["available_hours"] = [generate_available_hours() for _ in range(len(df))]  # Random working hours
    df["description"] = "Experienced in " + df["specialty"]  # Auto-generate based on specialty

    # Ensure no NaN values before generating embeddings
    df["description"] = df["description"].fillna("Unknown")

    # Generate embeddings and ensure correct alignment
    embeddings = generate_batch_embeddings(df["description"].tolist())

    # Ensure embeddings list matches the number of rows
    if len(embeddings) == len(df):
        df["embedding_vector"] = embeddings
    else:
        print("⚠️ Embedding length mismatch, using default vectors.")
        df["embedding_vector"] = [json.dumps([0.0] * 1536)] * len(df)

    return df

# Load Doctor dataset to get number of unique locations required
doctor_file = input_files["Doctor"]
df_doctor = pd.read_csv(doctor_file)
num_locations = df_doctor.shape[0]  # Number of unique locations should match number of doctors

# Ensure enough clinics are available by randomly selecting from the list if needed
clinics_extended = random.choices(clinics, k=num_locations) if num_locations > len(clinics) else random.sample(clinics, num_locations)

# Creating the Location DataFrame
location_data = []
for i, clinic in enumerate(clinics_extended):
    location_entry = {
        "locationId": i + 1,
        "clinicName": clinic["clinicName"],
        "Address": clinic["Address"],
        "postalCode": clinic["postalCode"],
        "medications": assign_services(medications_list),
        "procedures": assign_services(procedures_list)
    }
    location_data.append(location_entry)

location_df = pd.DataFrame(location_data)

# Save to CSV
location_output_path = os.path.join(output_folder, "Location.csv")
location_df.to_csv(location_output_path, index=False)
print(f"✅ Location data generated and saved to {location_output_path}")

# Define possible values for allergies and medical conditions
common_allergies = [
    "Penicillin", "Pollen", "Peanuts", "Shellfish", "Dust Mites", "Latex", "Soy", "Wheat", "Milk", "Eggs",
    "Insect Stings", "Mold", "Pet Dander", "NSAIDs", "Sulfa Drugs", "Nickel", "Fragrances", "Aspirin", "Alcohol", "Gluten"
]

common_conditions = [
    "Hypertension", "Diabetes", "Asthma", "Arthritis", "Heart Disease", "Epilepsy", "Migraine", "Thyroid Disorder",
    "Chronic Kidney Disease", "COPD", "Obesity", "Depression", "Anxiety", "High Cholesterol", "Osteoporosis", "Stroke",
    "Liver Disease", "Sleep Apnea", "Cancer", "IBS"
]

# Function to generate random allergies and conditions
def get_random_conditions(condition_list):
    """Randomly assigns 0, 1, or multiple conditions from the list."""
    num_conditions = random.choice([0, 1, random.randint(2, 4)])  # Choose 0, 1, or 2-4 conditions
    return "; ".join(random.sample(condition_list, num_conditions)) if num_conditions > 0 else "None"

# Function to clean Patient data
def clean_patient_data(df):
    """Format Patient data to match the suggested schema."""
    df.rename(columns={"PatientID": "patientId", "firstname": "FirstName", "lastname": "LastName"}, inplace=True)

    # Combine First and Last Name into a single field
    df["name"] = df["FirstName"] + " " + df["LastName"]
    df.drop(columns=["FirstName", "LastName"], inplace=True)

    # Add missing attributes
    df["age"] = [random.randint(18, 75) for _ in range(len(df))]  # Random age between 18-75
    df["gender"] = [random.choice(["Male", "Female"]) for _ in range(len(df))]  # Random gender
    df["drug_allergies"] = [get_random_conditions(common_allergies) for _ in range(len(df))]  # Random allergies
    df["medical_conditions"] = [get_random_conditions(common_conditions) for _ in range(len(df))]  # Random conditions

    return df

# Define cleaning functions
cleaning_functions = {
    "Doctor": clean_doctor_data,
    "Patient": clean_patient_data
}

# Process each dataset
for name, file in input_files.items():
    try:
        df = pd.read_csv(file)
        cleaned_df = cleaning_functions[name](df)

        # Drop any accidental extra columns before saving
        cleaned_df = cleaned_df.loc[:, ~cleaned_df.columns.str.contains("^Unnamed")]

        # Save cleaned file
        output_path = os.path.join(output_folder, f"{name}_Cleaned.csv")
        cleaned_df.to_csv(output_path, index=False)
        print(f"✅ {name} data cleaned and saved to {output_path}")

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")

print("\nDoctor and Patient datasets have been cleaned. Check the 'cleaned_data' folder for the output files.")

# Function to generate random timestamps within the past year and between 10 AM to 2 PM
def generate_random_timestamp():
    """Generate a random timestamp within the past year and between 10 AM to 2 PM."""
    now = datetime.now(timezone.utc)
    one_year_ago = now - timedelta(days=365)
    random_date = one_year_ago + timedelta(days=random.randint(0, 365))
    random_hour = random.randint(10, 13)  # 10 AM to 1 PM
    random_minute = random.randint(0, 59)
    random_second = random.randint(0, 59)
    random_timestamp = random_date.replace(hour=random_hour, minute=random_minute, second=random_second, microsecond=0)
    return random_timestamp.replace(tzinfo=None).strftime('%Y-%m-%d %H:%M:%S')

# Load cleaned Patient data
patient_file = os.path.join(output_folder, "Patient_Cleaned.csv")
df_patient = pd.read_csv(patient_file)

# Generate PatientChat, ChatMessage, and ChatResponse data
patient_chat_data = []
chat_message_data = []
chat_response_data = []

for index, row in df_patient.iterrows():
    patient_id = row["patientId"]
    patient_name = row["name"]
    allergies = row["drug_allergies"]
    conditions = row["medical_conditions"]

    # Create PatientChat entry
    chat_id = index + 1
    patient_chat_data.append({
        "chatId": chat_id,
        "patientId": patient_id,
        "Title": f"Chat with {patient_name}"
    })

    # Create ChatMessage entry
    message_id = index + 1
    chat_message_data.append({
        "messageId": message_id,
        "chatId": chat_id,
        "content": f"Patient {patient_name} has allergies: {allergies} and conditions: {conditions}",
        "timestamp": generate_random_timestamp()
    })

    # Create ChatResponse entry
    response_id = index + 1
    chat_response_data.append({
        "responseId": response_id,
        "chatId": chat_id,
        "content": f"Response to {patient_name} regarding allergies: {allergies} and conditions: {conditions}",
        "timestamp": generate_random_timestamp()
    })

# Convert to DataFrames
df_patient_chat = pd.DataFrame(patient_chat_data)
df_chat_message = pd.DataFrame(chat_message_data)
df_chat_response = pd.DataFrame(chat_response_data)

# Save each DataFrame to CSV files
patient_chat_output_path = os.path.join(output_folder, "PatientChat.csv")
df_patient_chat.to_csv(patient_chat_output_path, index=False, encoding="utf-8")

chat_message_output_path = os.path.join(output_folder, "ChatMessage.csv")
df_chat_message.to_csv(chat_message_output_path, index=False, encoding="utf-8")

chat_response_output_path = os.path.join(output_folder, "ChatResponse.csv")
df_chat_response.to_csv(chat_response_output_path, index=False, encoding="utf-8")

print(f"✅ Patient chat data generated and saved to {patient_chat_output_path}")
print(f"✅ Chat message data generated and saved to {chat_message_output_path}")
print(f"✅ Chat response data generated and saved to {chat_response_output_path}")

# Generate Admin data
admin_roles = [
    "System Administrator", "Database Administrator", "Network Administrator", "Security Administrator",
    "Application Administrator", "Web Administrator", "IT Support Specialist", "DevOps Engineer",
    "Cloud Administrator", "Technical Support Engineer"
]

admin_data = []
for i in range(10):
    admin_data.append({
        "adminId": i + 1,
        "name": f"admin{i + 1}",
        "admin_role": admin_roles[i]
    })

df_admin = pd.DataFrame(admin_data)

# Save Admin data to CSV
admin_output_path = os.path.join(output_folder, "Admin.csv")
df_admin.to_csv(admin_output_path, index=False)

print(f"✅ Admin data generated and saved to {admin_output_path}")