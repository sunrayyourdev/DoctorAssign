import apiClient from '../config';

// Types for chat messages
export interface UserMessage {
  content: string;
  timestamp: string;
}

export interface AssistantMessage {
  content: string;
  timestamp: string;
}

export interface ChatLog {
  chatId: number;
  patientId: number;
  Title: string;
  messages: UserMessage[];
  responses: AssistantMessage[];
  chat_timestamp: string;
}

// Chat service functions
export const chatService = {
  /**
   * Send a message to the chatbot
   * @param patientId The ID of the patient
   * @param content The message content
   * @returns The chatbot response
   */
  sendMessage: async (patientId: number, content: string) => {
    const response = await apiClient.post('/chatbot_response', {
      patientId,
      content,
    });
    return response.data;
  },

  /**
   * Get chat log for a specific patient
   * @param patientId The ID of the patient
   * @returns The patient's chat log
   */
  getChatLog: async (patientId: number) => {
    const response = await apiClient.get(`/get_chat_log/${patientId}`);
    return response.data.chat_log as ChatLog;
  },

  /**
   * Reset a patient's chat log
   * @param patientId The ID of the patient
   * @returns Confirmation of reset
   */
  resetChat: async (patientId: number) => {
    const response = await apiClient.post(`/resetPatientChat/${patientId}`);
    return response.data;
  },

  /**
   * Get patient chat for a specific doctor
   * @param doctorId The ID of the doctor
   * @param patientId The ID of the patient
   * @returns The patient's chat with symptoms
   */
  getPatientChat: async (doctorId: number, patientId: number) => {
    const response = await apiClient.get(`/get_patient_chat/${doctorId}/${patientId}`);
    return response.data;
  },

  /**
   * Get all patient chats for a doctor
   * @param doctorId The ID of the doctor
   * @returns All patient chats for the doctor
   */
  getAllPatientChats: async (doctorId: number) => {
    const response = await apiClient.get(`/get_all_patient_chats/${doctorId}`);
    return response.data.chat_logs as ChatLog[];
  }
};

export default chatService; 