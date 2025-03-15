import axios from 'axios';

// Base API configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5010';

// Create an Axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
  timeout: 10000, // 10 seconds timeout
});

// Request interceptor - useful for adding auth tokens, etc.
apiClient.interceptors.request.use(
  (config) => {
    // You can add authentication headers here if needed
    // For example: const token = localStorage.getItem('token');
    // if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor - global error handling
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Handle different error statuses here
    const { response } = error;
    
    if (response) {
      // Log errors for debugging
      console.error('API Error:', response.status, response.data);
      
      // Handle specific error codes
      switch (response.status) {
        case 401:
          // Handle unauthorized errors
          console.error('Unauthorized access');
          break;
        case 404:
          // Handle not found errors
          console.error('Resource not found');
          break;
        case 500:
          // Handle server errors
          console.error('Server error');
          break;
      }
    } else {
      console.error('Network Error:', error.message);
    }
    
    return Promise.reject(error);
  }
);

export default apiClient; 