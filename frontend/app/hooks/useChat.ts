import { useState, useCallback, useEffect } from 'react';
import { chatService, UserMessage, AssistantMessage } from '../api';

interface UseChatReturn {
  messages: UserMessage[];
  responses: AssistantMessage[];
  isLoading: boolean;
  error: string | null;
  sendMessage: (patientId: number, content: string) => Promise<void>;
  resetChat: (patientId: number) => Promise<void>;
}

export const useChat = (patientId?: number): UseChatReturn => {
  const [messages, setMessages] = useState<UserMessage[]>([]);
  const [responses, setResponses] = useState<AssistantMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load chat history from server if patientId is provided
  const loadChatHistory = useCallback(async () => {
    if (!patientId) return;
    
    try {
      setIsLoading(true);
      setError(null);
      
      const chatLog = await chatService.getChatLog(patientId);
      setMessages(chatLog.messages || []);
      setResponses(chatLog.responses || []);
    } catch (error) {
      console.error('Failed to load chat history:', error);
      setError('Failed to load chat history. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  }, [patientId]);

  // Send a message to the chatbot
  const sendMessage = useCallback(async (patientId: number, content: string) => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Add the message to the local state immediately for a responsive UI
      const newUserMessage: UserMessage = {
        content,
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev, newUserMessage]);
      
      // Send the message to the server
      const response = await chatService.sendMessage(patientId, content);
      
      // Update the responses with the chatbot's reply
      if (response.response) {
        const newResponse: AssistantMessage = {
          content: response.response,
          timestamp: new Date().toISOString()
        };
        
        setResponses(prev => [...prev, newResponse]);
      }
      
      // If there's a chat_history in the response, update the full conversation
      if (response.chat_history) {
        // Extract user messages and assistant responses from chat_history
        const userMessages: UserMessage[] = [];
        const assistantResponses: AssistantMessage[] = [];
        
        response.chat_history.forEach((item: any) => {
          if (item.role === 'user') {
            userMessages.push({
              content: item.content,
              timestamp: new Date().toISOString()
            });
          } else if (item.role === 'assistant') {
            assistantResponses.push({
              content: item.content,
              timestamp: new Date().toISOString()
            });
          }
        });
        
        setMessages(userMessages);
        setResponses(assistantResponses);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      setError('Failed to send message. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Reset the chat history
  const resetChat = useCallback(async (patientId: number) => {
    try {
      setIsLoading(true);
      setError(null);
      
      await chatService.resetChat(patientId);
      
      // Clear local state
      setMessages([]);
      setResponses([]);
    } catch (error) {
      console.error('Failed to reset chat:', error);
      setError('Failed to reset chat. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Load chat history on initialization if patientId is provided
  useEffect(() => {
    if (patientId) {
      loadChatHistory();
    }
  }, [patientId, loadChatHistory]);

  return {
    messages,
    responses,
    isLoading,
    error,
    sendMessage,
    resetChat
  };
};

export default useChat; 