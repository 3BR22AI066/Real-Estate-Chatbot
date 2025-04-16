import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // API endpoint - update this to match your Flask server URL
  const API_URL = 'http://localhost:5000/chat';

  // Scroll to bottom of chat
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Send message to backend
  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message to chat
    const userMessage = { text: input, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Send to Flask backend
      const response = await axios.post(API_URL, {
        message: input
      });

      // Add bot response to chat
      const botMessage = { 
        text: response.data.response, 
        sender: 'bot',
        data: response.data.data 
      };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = { 
        text: "Sorry, I encountered an error. Please try again.", 
        sender: 'bot' 
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Format property data into a nice display
  const renderProperty = (property) => {
    return (
      <div key={`${property.city}-${property.price}`} className="property-card">
        <h4>{property.city}, {property.state}</h4>
        <div className="property-details">
          <span>🏠 {property.bed} bed, {property.bath} bath</span>
          <span>📏 {property.house_size ? `${property.house_size.toLocaleString()} sqft` : 'Size not available'}</span>
          <span>💰 {property.price ? `$${property.price.toLocaleString()}` : 'Price not available'}</span>
        </div>
      </div>
    );
  };

  // Format message with properties
  const renderMessage = (message) => {
    if (message.sender === 'user') {
      return <div className="message user-message">{message.text}</div>;
    }

    // Check if message contains property data
    const hasProperties = message.data?.properties?.length > 0;

    return (
      <div className="message bot-message">
        <div>{message.text}</div>
        {hasProperties && (
          <div className="property-list">
            {message.data.properties.map(renderProperty)}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🏡 Real Estate Assistant</h1>
      </header>

      <div className="chat-container">
        <div className="messages">
          {messages.length === 0 ? (
            <div className="welcome-message">
              <p>Hello! I'm your real estate assistant. How can I help you today?</p>
              <p>Try asking:</p>
              <ul>
                <li>"Find me a 3 bedroom house under $300,000"</li>
                <li>"Show me 2 bed apartments in New York"</li>
                <li>"What's the market like in Miami?"</li>
              </ul>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div key={index} className={`message-container ${msg.sender}`}>
                {renderMessage(msg)}
              </div>
            ))
          )}
          {isLoading && (
            <div className="message-container bot">
              <div className="message bot-message">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={sendMessage} className="input-area">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message here..."
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading || !input.trim()}>
            {isLoading ? 'Sending...' : 'Send'}
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;