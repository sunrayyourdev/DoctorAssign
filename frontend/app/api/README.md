# API Integration Layer

This directory contains the API integration layer for connecting the frontend with the Flask backend.

## Architecture

- `config.ts`: Contains the Axios configuration including base URL, interceptors, and error handling
- `index.ts`: Exports API client and all services
- `services/`: Contains service modules for different API endpoints

### Services

- `chat.ts`: Handles chat-related API calls (sending messages, getting chat logs)
- `doctor.ts`: Handles doctor-related API calls (recommendations, approvals)
- `patient.ts`: Handles patient-related API calls (patient data, locations)

## Usage

```typescript
// Import services in your components/hooks
import { chatService, doctorService, patientService } from '@/app/api';

// Example: Sending a chat message
async function sendMessageToChat(patientId: number, message: string) {
  try {
    const response = await chatService.sendMessage(patientId, message);
    console.log('Response:', response);
  } catch (error) {
    console.error('Error sending message:', error);
  }
}

// Example: Getting doctor recommendations
async function getDoctorRecommendation(patientId: number) {
  try {
    const recommendation = await doctorService.getRecommendation(patientId);
    console.log('Recommended doctor:', recommendation.doctor_details);
    console.log('Analysis:', recommendation.chatgpt_analysis);
  } catch (error) {
    console.error('Error getting recommendation:', error);
  }
}
```

## Adding New Endpoints

To add a new endpoint:

1. Add the function to the appropriate service file
2. Export any new types from the service
3. Update the index.ts file to export any new types

## Error Handling

All API calls are wrapped with try/catch blocks and include proper error handling. The Axios interceptors in `config.ts` provide global error handling for all API calls.

## Environment Variables

The API base URL is configured using the environment variable `NEXT_PUBLIC_API_URL`. This allows for different environments (development, staging, production) to use different API endpoints.

Default configuration is in `.env.local` but can be overridden in environment-specific files like `.env.development` or `.env.production`. 