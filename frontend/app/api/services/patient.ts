import apiClient from '../config';

// Types for patient data
export interface Patient {
  patientId: number;
  email: string;
  name: string;
  age: number;
  gender: string;
  drug_allergies: string;
  medical_conditions: string;
}

// Patient service functions
export const patientService = {
  /**
   * Get all patients
   * @returns Array of all patients
   */
  getAllPatients: async () => {
    const response = await apiClient.post('/getall', {
      tableName: 'SQLUser.Patient'
    });
    return response.data.response as Patient[];
  },

  /**
   * Get patient by ID
   * @param patientId The ID of the patient
   * @returns Patient details
   */
  getPatientById: async (patientId: number) => {
    // Since there's no direct endpoint for getting a single patient,
    // we get all patients and filter by ID
    const response = await apiClient.post('/getall', {
      tableName: 'SQLUser.Patient'
    });
    
    const patients = response.data.response as Patient[];
    return patients.find(patient => patient.patientId === patientId);
  },

  /**
   * Insert a new patient
   * @param patient Patient data without ID (ID will be assigned by the server)
   * @returns Confirmation message
   */
  addPatient: async (patient: Omit<Patient, 'patientId'>) => {
    // Extract patient properties
    const { email, name, age, gender, drug_allergies, medical_conditions } = patient;
    
    const response = await apiClient.post('/insert', {
      tableName: 'SQLUser.Patient',
      columns: ['email', 'name', 'age', 'gender', 'drug_allergies', 'medical_conditions'],
      values: [[email, name, age, gender, drug_allergies, medical_conditions]]
    });
    
    return response.data;
  },

  /**
   * Get all locations (clinics)
   * @returns Array of all locations
   */
  getAllLocations: async () => {
    const response = await apiClient.post('/getall', {
      tableName: 'SQLUser.Location'
    });
    return response.data.response;
  }
};

export default patientService; 