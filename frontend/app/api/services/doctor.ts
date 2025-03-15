import apiClient from '../config';

// Types for doctor data
export interface Doctor {
  doctorId: number;
  name: string;
  specialty: string;
  doctorContact: string;
  locationId: number;
  experience_years: number;
  available_hours: string;
  description: string;
  embedding_vector?: string;
}

export interface DoctorRecommendation {
  doctor_details: Doctor;
  chatgpt_analysis: string;
}

// Interface for patient chats/appointments
export interface PatientChat {
  chatId: number;
  patientId: number;
  title: string;
  timestamp: string;
  messages: string[];
  responses: string[];
  patientName?: string;
  reason?: string;
  requestedTime?: string;
  status?: string;
}

// Doctor service functions
export const doctorService = {
  /**
   * Get all doctors
   * @returns Array of all doctors
   */
  getAllDoctors: async () => {
    const response = await apiClient.post('/getall', {
      tableName: 'SQLUser.Doctor'
    });
    return response.data.response as Doctor[];
  },

  /**
   * Get doctor by ID
   * @param doctorId The ID of the doctor
   * @returns Doctor details
   */
  getDoctorById: async (doctorId: number) => {
    // This is a mock implementation since there's no direct endpoint
    // Instead, we get all doctors and filter by ID
    const response = await apiClient.post('/getall', {
      tableName: 'SQLUser.Doctor'
    });
    
    const doctors = response.data.response as Doctor[];
    return doctors.find(doctor => doctor.doctorId === doctorId);
  },

  /**
   * Get doctor recommendation for a patient
   * @param patientId The ID of the patient
   * @returns Doctor recommendation with AI analysis
   */
  getRecommendation: async (patientId: number) => {
    const response = await apiClient.post('/recommend_doctor', {
      patientId
    });
    return response.data as DoctorRecommendation;
  },

  /**
   * Update doctor approval status
   * @param chatId The ID of the chat
   * @param approval Approval status (1 for approved, 0 for not approved)
   * @returns Confirmation message
   */
  updateDoctorApproval: async (chatId: number, approval: number) => {
    const response = await apiClient.post(`/update_doctor_approval/${chatId}`, {
      approval
    });
    return response.data;
  },

  /**
   * Get all patient chats/appointments for a doctor
   * @param doctorId The ID of the doctor
   * @returns Array of patient chats/appointments
   */
  getDoctorAppointments: async (doctorId: number) => {
    try {
      const response = await apiClient.get(`/get_all_patient_chats/${doctorId}`);
      return response.data.chat_logs as PatientChat[];
    } catch (error) {
      console.error('Error fetching doctor appointments:', error);
      return [];
    }
  }
};

export default doctorService; 