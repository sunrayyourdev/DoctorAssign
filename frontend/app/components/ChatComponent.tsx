import React, { useState } from 'react';
import useChat from '../hooks/useChat';

interface ChatComponentProps {
  patientId: number;
}

const ChatComponent: React.FC<ChatComponentProps> = ({ patientId }) => {
  const [message, setMessage] = useState('');
  const { messages, responses, isLoading, error, sendMessage, resetChat } = useChat(patientId);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!message.trim()) return;
    
    await sendMessage(patientId, message.trim());
    setMessage('');
  };

  const handleResetChat = async () => {
    await resetChat(patientId);
  };

  // Combine messages and responses into a chronological chat display
  const combinedChat = [...messages.map(msg => ({ ...msg, role: 'user' })), 
                        ...responses.map(resp => ({ ...resp, role: 'assistant' }))]
    .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto">
      {/* Chat header */}
      <div className="bg-white p-4 shadow-sm rounded-t-lg flex justify-between items-center">
        <h2 className="text-xl font-semibold text-gray-800">Patient Chat</h2>
        <button 
          onClick={handleResetChat}
          className="px-3 py-1 bg-red-100 text-red-700 text-sm font-medium rounded hover:bg-red-200 transition-colors"
          aria-label="Reset chat history"
        >
          Reset Chat
        </button>
      </div>
      
      {/* Chat messages */}
      <div className="flex-1 overflow-y-auto p-4 bg-gray-50 space-y-4">
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4" role="alert">
            <p>{error}</p>
          </div>
        )}
        
        {combinedChat.length === 0 ? (
          <div className="text-center text-gray-500 py-10">
            <p>No messages yet. Start a conversation!</p>
          </div>
        ) : (
          combinedChat.map((item, index) => (
            <div 
              key={index} 
              className={`flex ${item.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div 
                className={`max-w-[80%] px-4 py-2 rounded-lg ${
                  item.role === 'user' 
                    ? 'bg-blue-600 text-white rounded-br-none' 
                    : 'bg-white text-gray-800 rounded-bl-none shadow-sm'
                }`}
              >
                <p>{item.content}</p>
                <div className={`text-xs mt-1 ${item.role === 'user' ? 'text-blue-200' : 'text-gray-500'}`}>
                  {new Date(item.timestamp).toLocaleTimeString()}
                </div>
              </div>
            </div>
          ))
        )}
        
        {isLoading && (
          <div className="flex justify-center py-2">
            <div className="animate-pulse flex space-x-2">
              <div className="w-3 h-3 bg-gray-400 rounded-full"></div>
              <div className="w-3 h-3 bg-gray-400 rounded-full"></div>
              <div className="w-3 h-3 bg-gray-400 rounded-full"></div>
            </div>
          </div>
        )}
      </div>
      
      {/* Message input */}
      <form onSubmit={handleSendMessage} className="bg-white p-4 shadow-sm rounded-b-lg">
        <div className="flex space-x-2">
          <input
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isLoading}
            aria-label="Message input"
          />
          <button
            type="submit"
            className="px-4 py-2 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={isLoading || !message.trim()}
            aria-label="Send message"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatComponent; 