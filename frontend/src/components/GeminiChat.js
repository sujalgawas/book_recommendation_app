import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';

// Configure axios instance
const apiClient = axios.create({
  baseURL: 'http://localhost:5000',
  withCredentials: true,
});

const GeminiChat = () => {
  const [query, setQuery] = useState('');
  const [conversation, setConversation] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const { userData } = useAuth();
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Scroll to bottom whenever conversation updates
  useEffect(() => {
    scrollToBottom();
  }, [conversation]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleQueryChange = (event) => {
    setQuery(event.target.value);
  };

  const handleSubmitQuery = async (event) => {
    event.preventDefault();
    
    if (!query.trim()) {
      setError('Please enter a question.');
      return;
    }
    
    if (!userData || !userData.id) {
      setError('Could not identify user. Please log in again.');
      return;
    }

    // Add user message to conversation
    const userMessage = {
      role: 'user',
      content: query,
      timestamp: new Date().toISOString()
    };
    
    setConversation(prev => [...prev, userMessage]);
    setIsLoading(true);
    setError('');
    setQuery(''); // Clear input field after sending
    
    try {
      const response = await apiClient.post(`/api/user/${userData.id}/ask_gemini`, {
        query: userMessage.content,
        // Optional: Send conversation history for context
        conversation: conversation.slice(-6) // Last 6 messages for context
      });
      
      if (response.data && response.data.answer) {
        // Add assistant message to conversation
        const assistantMessage = {
          role: 'assistant',
          content: response.data.answer,
          timestamp: new Date().toISOString()
        };
        
        setConversation(prev => [...prev, assistantMessage]);
      } else if (response.data && response.data.error) {
        setError(response.data.error);
      } else {
        throw new Error("Invalid response received from server.");
      }
    } catch (err) {
      console.error("Error fetching Gemini response:", err.response || err);
      setError(err.response?.data?.error || err.message || 'An error occurred while fetching the answer.');
    } finally {
      setIsLoading(false);
      // Focus back on input after response
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (event) => {
    // Submit on Enter (without Shift)
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSubmitQuery(event);
    }
  };

  const clearConversation = () => {
    setConversation([]);
    setError('');
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h2>Gemini Book Assistant</h2>
        {conversation.length > 0 && (
          <button 
            className="clear-button" 
            onClick={clearConversation}
            aria-label="Clear conversation"
          >
            Clear Chat
          </button>
        )}
      </div>

      <div className="messages-container">
        {conversation.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">💬</div>
            <h3>Ask Gemini about your books</h3>
            <p>Try asking about book recommendations, summaries, or authors in your collection.</p>
            <div className="suggestion-chips">
              <button onClick={() => setQuery("Recommend a fantasy book from my list")}>
                Recommend a fantasy book
              </button>
              <button onClick={() => setQuery("What's the newest book in my collection?")}>
                Newest book
              </button>
              <button onClick={() => setQuery("Who's my most frequent author?")}>
                Most frequent author
              </button>
            </div>
          </div>
        ) : (
          <>
            {conversation.map((message, index) => (
              <div 
                key={index} 
                className={`message ${message.role === 'user' ? 'user-message' : 'assistant-message'}`}
              >
                <div className="message-content">
                  {message.content}
                </div>
                <div className="message-timestamp">
                  {new Date(message.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                </div>
              </div>
            ))}
          </>
        )}
        
        {isLoading && (
          <div className="message assistant-message">
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {error && <div className="error-message">Error: {error}</div>}

      <form onSubmit={handleSubmitQuery} className="chat-input-form">
        <textarea
          ref={inputRef}
          value={query}
          onChange={handleQueryChange}
          onKeyDown={handleKeyDown}
          placeholder="Ask about your books..."
          disabled={isLoading}
          rows={1}
          className="chat-input"
        />
        <button 
          type="submit" 
          disabled={isLoading || !query.trim()} 
          className="send-button"
          aria-label="Send message"
        >
          {isLoading ? 
            <span className="loading-spinner"></span> : 
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M22 2L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          }
        </button>
      </form>

      <style jsx>{`
        .chat-container {
          display: flex;
          flex-direction: column;
          height: 720px;
          max-heigh: 100%;
          width: 100%;
          background-color: #121212;
          border-radius: 12px;
          overflow: hidden;
          box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        }
        
        .chat-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 16px 20px;
          background-color: #1e1e2d;
          border-bottom: 1px solid #2a2a3a;
        }
        
        .chat-header h2 {
          margin: 0;
          color: #e0e0e0;
          font-size: 1.25rem;
          font-weight: 600;
        }
        
        .clear-button {
          background-color: transparent;
          color: #9ca3af;
          border: 1px solid #424250;
          border-radius: 6px;
          padding: 6px 12px;
          font-size: 0.875rem;
          cursor: pointer;
          transition: all 0.2s ease;
        }
        
        .clear-button:hover {
          background-color: #2a2a3a;
          color: #e0e0e0;
        }
        
        .messages-container {
          flex: 1;
          overflow-y: auto;
          padding: 20px;
          display: flex;
          flex-direction: column;
          gap: 16px;
          scroll-behavior: smooth;
        }
        
        .message {
          display: flex;
          flex-direction: column;
          max-width: 80%;
          padding: 12px 16px;
          border-radius: 12px;
          animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        
        .user-message {
          align-self: flex-end;
          background-color: #2563eb;
          color: white;
          border-bottom-right-radius: 4px;
        }
        
        .assistant-message {
          align-self: flex-start;
          background-color: #2a2a3a;
          color: #e0e0e0;
          border-bottom-left-radius: 4px;
        }
        
        .message-content {
          white-space: pre-wrap;
          word-break: break-word;
          line-height: 1.5;
        }
        
        .message-timestamp {
          font-size: 0.7rem;
          margin-top: 6px;
          align-self: flex-end;
          opacity: 0.7;
        }
        
        .typing-indicator {
          display: flex;
          align-items: center;
          gap: 4px;
        }
        
        .typing-indicator span {
          width: 8px;
          height: 8px;
          background-color: #9ca3af;
          border-radius: 50%;
          animation: typing 1.4s infinite ease-in-out;
        }
        
        .typing-indicator span:nth-child(1) {
          animation-delay: 0s;
        }
        
        .typing-indicator span:nth-child(2) {
          animation-delay: 0.2s;
        }
        
        .typing-indicator span:nth-child(3) {
          animation-delay: 0.4s;
        }
        
        @keyframes typing {
          0%, 60%, 100% { transform: translateY(0); }
          30% { transform: translateY(-6px); }
        }
        
        .chat-input-form {
          display: flex;
          align-items: center;
          padding: 16px;
          background-color: #1a1a2a;
          border-top: 1px solid #2a2a3a;
        }
        
        .chat-input {
          flex: 1;
          padding: 12px 16px;
          border-radius: 8px;
          border: 1px solid #3a3a4a;
          background-color: #202030;
          color: #e0e0e0;
          font-size: 1rem;
          resize: none;
          outline: none;
          transition: border-color 0.2s ease;
          max-height: 120px;
          overflow-y: auto;
        }
        
        .chat-input:focus {
          border-color: #4a4a6a;
          box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
        }
        
        .chat-input::placeholder {
          color: #9ca3af;
        }
        
        .send-button {
          display: flex;
          justify-content: center;
          align-items: center;
          width: 40px;
          height: 40px;
          margin-left: 12px;
          border-radius: 50%;
          background-color: #2563eb;
          color: white;
          border: none;
          cursor: pointer;
          transition: background-color 0.2s ease;
        }
        
        .send-button:hover {
          background-color: #1d4ed8;
        }
        
        .send-button:disabled {
          background-color: #3a3a4a;
          cursor: not-allowed;
        }
        
        .loading-spinner {
          width: 20px;
          height: 20px;
          border: 2px solid rgba(255, 255, 255, 0.3);
          border-radius: 50%;
          border-top-color: white;
          animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        
        .error-message {
          padding: 10px 16px;
          margin: 0 16px;
          background-color: rgba(220, 38, 38, 0.1);
          color: #ef4444;
          border-radius: 6px;
          font-size: 0.875rem;
          border-left: 3px solid #ef4444;
        }
        
        .empty-state {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100%;
          text-align: center;
          color: #9ca3af;
          padding: 20px;
        }
        
        .empty-icon {
          font-size: 3rem;
          margin-bottom: 16px;
        }
        
        .empty-state h3 {
          margin: 0 0 8px;
          color: #e0e0e0;
        }
        
        .empty-state p {
          margin: 0 0 24px;
          max-width: 80%;
        }
        
        .suggestion-chips {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          justify-content: center;
          width: 100%;
          max-width: 500px;
        }
        
        .suggestion-chips button {
          background-color: #2a2a3a;
          color: #e0e0e0;
          border: none;
          border-radius: 16px;
          padding: 8px 16px;
          font-size: 0.875rem;
          cursor: pointer;
          transition: background-color 0.2s ease;
          white-space: nowrap;
        }
        
        .suggestion-chips button:hover {
          background-color: #3a3a4a;
        }
        
        /* Custom scrollbar */
        .messages-container::-webkit-scrollbar {
          width: 6px;
        }
        
        .messages-container::-webkit-scrollbar-track {
          background: #1a1a2a;
        }
        
        .messages-container::-webkit-scrollbar-thumb {
          background-color: #3a3a4a;
          border-radius: 6px;
        }
        
        .messages-container::-webkit-scrollbar-thumb:hover {
          background-color: #4a4a5a;
        }
      `}</style>
    </div>
  );
};

export default GeminiChat;