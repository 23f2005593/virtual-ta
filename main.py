from fastapi import FastAPI, HTTPException
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import json
from pydantic import BaseModel
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message
from typing import Optional
from dotenv import load_dotenv
import uvicorn
import os

load_dotenv() 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('pinecone_api_key'))
assistant = pc.assistant.Assistant(assistant_name="tds-virtual-assistant")

# Request model
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

# Response model (matches expected format)
class QueryResponse(BaseModel):
    answer: str
    links: list[dict[str, str]]

# HTML template for the frontend
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TDS Virtual Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background-color: #2C2F33;
            color: #ffffff;
            height: 100vh;
            overflow: hidden;
        }

        .chat-container {
            display: flex;
            flex-direction: column;
            height: 100vh;
            max-width: 768px;
            margin: 0 auto;
            background-color: #2C2F33;
        }

        .header {
            padding: 20px;
            border-bottom: 1px solid #262626;
            background-color: #2C2F33;
        }

        .header h1 {
            font-size: 20px;
            font-weight: 600;
            color: #ffffff;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 0;
            background-color: #2C2F33;
        }

        .chat-messages::-webkit-scrollbar {
            width: 4px;
        }

        .chat-messages::-webkit-scrollbar-track {
            background: transparent;
        }

        .chat-messages::-webkit-scrollbar-thumb {
            background: #333333;
            border-radius: 2px;
        }

        .message {
            padding: 24px 20px;
            border-bottom: 1px solid #0a0a0a;
        }

        .message.user {
            background-color: #2C2F33;
        }

        .message.assistant {
            background-color: #0a0a0a;
        }

        .message-header {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            gap: 8px;
        }

        .avatar {
            width: 24px;
            height: 24px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            font-weight: 600;
        }

        .user-avatar {
            background-color: #1d4ed8;
            color: white;
        }

        .assistant-avatar {
            background-color: #ffffff;
            color: #000000;
        }

        .message-author {
            font-size: 14px;
            font-weight: 600;
            color: #ffffff;
        }

        .message-content {
            font-size: 15px;
            line-height: 1.6;
            color: #ffffff;
            margin-left: 32px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .links-section {
            margin-left: 32px;
            margin-top: 16px;
        }

        .links-title {
            font-size: 14px;
            font-weight: 600;
            color: #9ca3af;
            margin-bottom: 8px;
        }

        .link-item {
            display: block;
            padding: 8px 12px;
            margin-bottom: 4px;
            background-color: #1a1a1a;
            border: 1px solid #262626;
            border-radius: 6px;
            text-decoration: none;
            color: #ffffff;
            font-size: 14px;
            transition: background-color 0.2s ease;
        }

        .link-item:hover {
            background-color: #262626;
        }

        .input-section {
            padding: 20px;
            border-top: 1px solid #262626;
            background-color: #2C2F33;
        }

        .input-container {
            position: relative;
            background-color: #1a1a1a;
            border: 1px solid #262626;
            border-radius: 24px;
            padding: 4px;
            display: flex;
            align-items: flex-end;
            gap: 8px;
            min-height: 48px;
        }

        .input-container:focus-within {
            border-color: #404040;
        }

        .image-upload-wrapper {
            display: flex;
            align-items: center;
            padding-left: 8px;
        }

        .image-upload {
            position: relative;
            overflow: hidden;
        }

        .image-upload input[type=file] {
            position: absolute;
            opacity: 0;
            width: 100%;
            height: 100%;
            cursor: pointer;
        }

        .image-upload-button {
            width: 32px;
            height: 32px;
            border-radius: 6px;
            background-color: transparent;
            border: none;
            color: #9ca3af;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            transition: color 0.2s ease;
        }

        .image-upload-button:hover {
            color: #ffffff;
        }

        .message-input {
            flex: 1;
            border: none;
            background: transparent;
            color: #ffffff;
            font-size: 15px;
            line-height: 1.4;
            padding: 12px 8px;
            resize: none;
            min-height: 20px;
            max-height: 120px;
            outline: none;
        }

        .message-input::placeholder {
            color: #6b7280;
        }

        .send-button {
            width: 32px;
            height: 32px;
            border-radius: 6px;
            background-color: #ffffff;
            border: none;
            color: #000000;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            font-weight: 600;
            margin-right: 8px;
            transition: background-color 0.2s ease;
        }

        .send-button:hover:not(:disabled) {
            background-color: #e5e5e5;
        }

        .send-button:disabled {
            background-color: #404040;
            color: #6b7280;
            cursor: not-allowed;
        }

        .image-preview {
            max-width: 200px;
            max-height: 150px;
            border-radius: 8px;
            margin: 8px 0;
            border: 1px solid #262626;
        }

        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            padding: 40px 20px;
            text-align: center;
        }

        .empty-state-icon {
            width: 48px;
            height: 48px;
            background-color: #1a1a1a;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            margin-bottom: 16px;
        }

        .empty-state h2 {
            font-size: 20px;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 8px;
        }

        .empty-state p {
            font-size: 15px;
            color: #6b7280;
            max-width: 400px;
        }

        .typing-indicator {
            display: flex;
            align-items: center;
            gap: 4px;
            margin-left: 32px;
            padding: 8px 0;
        }

        .typing-dot {
            width: 4px;
            height: 4px;
            border-radius: 50%;
            background-color: #6b7280;
            animation: typing 1.4s infinite ease-in-out;
        }

        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }

        @keyframes typing {
            0%, 80%, 100% { opacity: 0.3; }
            40% { opacity: 1; }
        }

        .loading-spinner {
            width: 16px;
            height: 16px;
            border: 2px solid #6b7280;
            border-top: 2px solid #000000;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Mobile Responsive */
        @media (max-width: 768px) {
            .header {
                padding: 16px;
            }

            .header h1 {
                font-size: 18px;
            }

            .message {
                padding: 20px 16px;
            }

            .input-section {
                padding: 16px;
            }

            .empty-state {
                padding: 32px 16px;
            }

            .empty-state h2 {
                font-size: 18px;
            }

            .empty-state p {
                font-size: 14px;
            }
        }

        @media (max-width: 480px) {
            .chat-container {
                max-width: 100%;
            }

            .header {
                padding: 12px 16px;
            }

            .message {
                padding: 16px 12px;
            }

            .message-content {
                margin-left: 28px;
            }

            .links-section {
                margin-left: 28px;
            }

            .typing-indicator {
                margin-left: 28px;
            }

            .input-section {
                padding: 12px;
            }

            .avatar {
                width: 20px;
                height: 20px;
                font-size: 12px;
            }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="header">
            <h1>TDS Virtual Assistant</h1>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="empty-state">
                <div class="empty-state-icon">🤖</div>
                <h2>Welcome to TDS Assistant</h2>
                <p>Ask me anything or upload an image to get started. I'm here to help!</p>
            </div>
        </div>
        
        <div class="input-section">
            <div class="input-container">
                <div class="image-upload-wrapper">
                    <div class="image-upload">
                        <input type="file" id="imageInput" accept="image/*">
                        <button type="button" class="image-upload-button">📎</button>
                    </div>
                </div>
                <textarea 
                    id="messageInput" 
                    class="message-input" 
                    placeholder="Message TDS Assistant..."
                    rows="1"
                ></textarea>
                <button id="sendButton" class="send-button">↑</button>
            </div>
            <img id="imagePreview" class="image-preview" style="display: none;">
        </div>
    </div>

    <script>
        const chatMessages = document.getElementById('chatMessages');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const imageInput = document.getElementById('imageInput');
        const imagePreview = document.getElementById('imagePreview');
        
        let currentImage = null;
        
        // Auto-resize textarea
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });
        
        // Handle image upload
        imageInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    currentImage = e.target.result.split(',')[1];
                    imagePreview.src = e.target.result;
                    imagePreview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });
        
        // Send message on Enter
        messageInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        sendButton.addEventListener('click', sendMessage);
        
        function clearEmptyState() {
            const emptyState = chatMessages.querySelector('.empty-state');
            if (emptyState) {
                emptyState.remove();
            }
        }
        
        function addMessage(content, isUser = false, links = []) {
            clearEmptyState();
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
            
            let messageHTML = `
                <div class="message-header">
                    <div class="avatar ${isUser ? 'user-avatar' : 'assistant-avatar'}">
                        ${isUser ? 'U' : 'AI'}
                    </div>
                    <span class="message-author">${isUser ? 'You' : 'TDS Assistant'}</span>
                </div>
                <div class="message-content">${content}</div>
            `;
            
            if (Array.isArray(links) && links.length > 0) {
                messageHTML += `
                    <div class="links-section">
                        <div class="links-title">Related Links</div>
                        ${links.map(link => `
                            <a href="${link.url}" target="_blank" class="link-item">
                                ${link.url.length > 50 ? link.url.substring(0, 47) + '...' : link.url}
                            </a>
                        `).join('')}
                    </div>
                `;
        }
            
            messageDiv.innerHTML = messageHTML;
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function addTypingIndicator() {
            clearEmptyState();
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message assistant';
            messageDiv.id = 'typing-indicator';
            messageDiv.innerHTML = `
                <div class="message-header">
                    <div class="avatar assistant-avatar">AI</div>
                    <span class="message-author">TDS Assistant</span>
                </div>
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            `;
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function removeTypingIndicator() {
            const typingIndicator = document.getElementById('typing-indicator');
            if (typingIndicator) {
                typingIndicator.remove();
            }
        }
        
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message && !currentImage) return;
            
            // Add user message
            if (message) {
                addMessage(message, true);
            }
            
            // Clear input
            messageInput.value = '';
            messageInput.style.height = 'auto';
            
            // Disable send button and show loading
            sendButton.disabled = true;
            sendButton.innerHTML = '<div class="loading-spinner"></div>';
            
            // Add typing indicator
            addTypingIndicator();
            
            try {
                const requestBody = {
                    question: message || "Please analyze this image"
                };
                
                if (currentImage) {
                    requestBody.image = currentImage;
                }
                
                // Use relative URL since we're serving from same domain
                const response = await fetch('/api/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(requestBody)
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const data = await response.json();
                
                // Remove typing indicator
                removeTypingIndicator();
                
                // Add assistant response
                addMessage(data.answer, false, data.links);
                
            } catch (error) {
                removeTypingIndicator();
                addMessage(`Sorry, I encountered an error: ${error.message}`, false);
            } finally {
                // Re-enable send button
                sendButton.disabled = false;
                sendButton.innerHTML = '↑';
                
                // Clear image
                currentImage = null;
                imagePreview.style.display = 'none';
                imageInput.value = '';
                
                // Focus back to input
                messageInput.focus();
            }
        }
        
        // Focus on input when page loads
        window.addEventListener('DOMContentLoaded', () => {
            messageInput.focus();
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the chat interface at the root URL"""
    return HTML_TEMPLATE

@app.post("/api/", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        # Prepare message for Pinecone assistant
        message = {
            "role": "user",
            "content": request.question
        }
        
        if request.image:
            message["image"] = {
                "data": request.image
            }
        
        # Get response from Pinecone assistant
        resp = assistant.chat(messages=[message])
        
        # Assume response is already in the correct format due to system prompt
        response_data = resp["message"]["content"]
        
        # If response is a string, try to parse it as JSON (in case Pinecone returns JSON as string)
        if isinstance(response_data, str):
            response_data = json.loads(response_data)
        
        return QueryResponse(**response_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/api/test")
async def test(): 
    return {"response": "Test Done"}
