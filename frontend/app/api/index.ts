import apiClient from './config';
import chatService from './services/chat';
import doctorService from './services/doctor';
import patientService from './services/patient';

// Export the API client
export { apiClient };

// Export all services
export { chatService, doctorService, patientService };

// Export types from services
export type { UserMessage, AssistantMessage, ChatLog } from './services/chat';
export type { Doctor, DoctorRecommendation, PatientChat } from './services/doctor';
export type { Patient } from './services/patient';

// Services will be exported here as they're created
// export * from './services/chat';
// export * from './services/doctor';
// export * from './services/patient'; 