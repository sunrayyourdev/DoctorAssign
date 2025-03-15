"use client";

import React, { useState, useEffect } from 'react';
import { apiClient } from '../api';

const ApiConnectionTest: React.FC = () => {
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [message, setMessage] = useState<string>('Testing connection to backend...');
  const [responseTime, setResponseTime] = useState<number | null>(null);

  useEffect(() => {
    const testConnection = async () => {
      const startTime = Date.now();
      try {
        // Simple ping to the backend root endpoint
        const response = await apiClient.get('/');
        const endTime = Date.now();
        setResponseTime(endTime - startTime);
        setStatus('success');
        setMessage('Successfully connected to the backend!');
      } catch (error) {
        setStatus('error');
        setMessage('Failed to connect to the backend. Make sure the server is running at http://localhost:5010');
        console.error('Backend connection error:', error);
      }
    };

    testConnection();
  }, []);

  return (
    <div className="p-4 rounded-lg max-w-xl mx-auto">
      <div className={`p-4 rounded-lg ${
        status === 'loading' ? 'bg-blue-50 border border-blue-200' :
        status === 'success' ? 'bg-green-50 border border-green-200' :
        'bg-red-50 border border-red-200'
      }`}>
        <h3 className={`text-lg font-semibold mb-2 ${
          status === 'loading' ? 'text-blue-700' :
          status === 'success' ? 'text-green-700' :
          'text-red-700'
        }`}>
          Backend API Connection Status
        </h3>
        
        <div className="flex items-center mb-2">
          <div className={`h-3 w-3 rounded-full mr-2 ${
            status === 'loading' ? 'bg-blue-500 animate-pulse' :
            status === 'success' ? 'bg-green-500' :
            'bg-red-500'
          }`} />
          <span className={
            status === 'loading' ? 'text-blue-600' :
            status === 'success' ? 'text-green-600' :
            'text-red-600'
          }>
            {status === 'loading' ? 'Connecting...' : 
             status === 'success' ? 'Connected' : 
             'Connection Failed'}
          </span>
        </div>
        
        <p className="text-gray-700">{message}</p>
        
        {responseTime !== null && status === 'success' && (
          <p className="text-sm text-gray-600 mt-2">
            Response time: {responseTime}ms
          </p>
        )}
      </div>
    </div>
  );
};

export default ApiConnectionTest; 