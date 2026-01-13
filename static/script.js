/**
 * Chatbot Frontend JavaScript Logic
 * Handles message sending, API communication, and UI interactions
 */

// Constants
const API_BASE_URL = '/api';
const CHAT_MESSAGES_CONTAINER = document.getElementById('chatMessages');
const MESSAGE_INPUT = document.getElementById('messageInput');
const SEND_BUTTON = document.getElementById('sendButton');
const MENU_BUTTON = document.getElementById('menuButton');
const SIDEBAR = document.getElementById('sidebar');
const SIDEBAR_OVERLAY = document.getElementById('sidebarOverlay');

// State
let isLoading = false;
let chatHistory = [];

/**
 * Initialize the chatbot
 */
document.addEventListener('DOMContentLoaded', () => {
    // Event listeners
    SEND_BUTTON.addEventListener('click', sendMessage);
    MESSAGE_INPUT.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !isLoading) {
            sendMessage();
        }
    });
    MENU_BUTTON.addEventListener('click', openSidebar);
    
    // Load chat history if available
    loadChatHistory();
});

/**
 * Send a message to the chatbot
 */
async function sendMessage() {
    const message = MESSAGE_INPUT.value.trim();
    
    if (!message || isLoading) {
        return;
    }
    
    // Disable input while processing
    isLoading = true;
    SEND_BUTTON.disabled = true;
    MESSAGE_INPUT.disabled = true;
    
    // Remove welcome container if present
    const welcomeContainer = document.querySelector('.welcome-container');
    if (welcomeContainer) {
        welcomeContainer.remove();
    }
    
    // Add user message to chat
    addMessageToChat('user', message);
    MESSAGE_INPUT.value = '';
    
    // Add to history
    chatHistory.push({ role: 'user', message: message });
    saveChatHistory();
    
    try {
        // Send message to backend
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            addMessageToChat('assistant', `Error: ${data.error}`);
        } else {
            // Add assistant response
            const assistantMessage = formatAssistantResponse(data);
            addMessageToChat('assistant', assistantMessage);
            
            // Add to history
            chatHistory.push({ 
                role: 'assistant', 
                message: data.answer,
                sources: data.sources 
            });
            saveChatHistory();
        }
    } catch (error) {
        console.error('Error:', error);
        addMessageToChat('assistant', 
            `I encountered an error while processing your request. Please try again. (${error.message})`
        );
    } finally {
        // Re-enable input
        isLoading = false;
        SEND_BUTTON.disabled = false;
        MESSAGE_INPUT.disabled = false;
        MESSAGE_INPUT.focus();
    }
}

/**
 * Add a message to the chat display
 */
function addMessageToChat(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (role === 'assistant') {
        // Content is already formatted HTML
        contentDiv.innerHTML = content;
    } else {
        // User message - escape HTML and wrap in paragraph
        contentDiv.innerHTML = `<p>${escapeHtml(content)}</p>`;
    }
    
    messageDiv.appendChild(contentDiv);
    CHAT_MESSAGES_CONTAINER.appendChild(messageDiv);
    
    // Scroll to bottom
    CHAT_MESSAGES_CONTAINER.scrollTop = CHAT_MESSAGES_CONTAINER.scrollHeight;
}

/**
 * Format the assistant response with sources
 */
function formatAssistantResponse(data) {
    let html = `<p>${escapeHtml(data.answer)}</p>`;
    
    if (data.sources && data.sources.length > 0) {
        html += '<div class="sources-section">';
        html += '<h4><i class="fas fa-file-alt"></i> Sources</h4>';
        html += '<div class="sources-list">';
        
        data.sources.forEach((source, index) => {
            html += `
                <div class="source-item">
                    <div class="source-number">${index + 1}</div>
                    <div class="source-content">
                        <h5>${escapeHtml(source.title)}</h5>
                        <p>${escapeHtml(source.snippet)}</p>
                    </div>
                </div>
            `;
        });
        
        html += '</div></div>';
    }
    
    return html;
}

/**
 * Set a query from example buttons
 */
function setQuery(query) {
    MESSAGE_INPUT.value = query;
    MESSAGE_INPUT.focus();
}

/**
 * Sidebar functions
 */
function openSidebar() {
    SIDEBAR.classList.add('active');
    SIDEBAR_OVERLAY.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeSidebar() {
    SIDEBAR.classList.remove('active');
    SIDEBAR_OVERLAY.classList.remove('active');
    document.body.style.overflow = 'auto';
}

/**
 * Clear chat history
 */
function clearChat() {
    chatHistory = [];
    localStorage.removeItem('chatHistory');
    CHAT_MESSAGES_CONTAINER.innerHTML = `
        <div class="welcome-container">
            <div class="welcome-icon">
                <i class="fas fa-book-open"></i>
            </div>
            <h2>Welcome to the Utah Digital Newspapers Chatbot</h2>
            <p>Ask questions about historical articles from Utah's digital newspaper archive.</p>
            
            <div class="example-queries">
                <p class="examples-title">Try asking:</p>
                <div class="example-buttons">
                    <button class="example-btn" onclick="setQuery('What articles mention women\'s suffrage in Utah?')">
                        <i class="fas fa-search"></i>
                        Women's Suffrage
                    </button>
                    <button class="example-btn" onclick="setQuery('Find articles about the 1918 Spanish flu in Salt Lake City')">
                        <i class="fas fa-virus"></i>
                        Spanish Flu 1918
                    </button>
                    <button class="example-btn" onclick="setQuery('Show me mining related articles from the 1900s')">
                        <i class="fas fa-pickaxe"></i>
                        Mining Articles
                    </button>
                    <button class="example-btn" onclick="setQuery('What newspapers covered the first transcontinental railroad?')">
                        <i class="fas fa-train"></i>
                        Railroad History
                    </button>
                </div>
            </div>
        </div>
    `;
    closeSidebar();
}

/**
 * Save chat history to localStorage
 */
function saveChatHistory() {
    try {
        localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
    } catch (e) {
        console.warn('Could not save chat history:', e);
    }
}

/**
 * Load chat history from localStorage
 */
function loadChatHistory() {
    try {
        const saved = localStorage.getItem('chatHistory');
        if (saved) {
            chatHistory = JSON.parse(saved);
            
            // Rebuild the chat display
            const welcomeContainer = document.querySelector('.welcome-container');
            if (welcomeContainer) {
                welcomeContainer.remove();
            }
            
            chatHistory.forEach(item => {
                if (item.role === 'user') {
                    addMessageToChat('user', item.message);
                } else {
                    const response = {
                        answer: item.message,
                        sources: item.sources || []
                    };
                    addMessageToChat('assistant', formatAssistantResponse(response));
                }
            });
        }
    } catch (e) {
        console.warn('Could not load chat history:', e);
    }
}

/**
 * Escape HTML special characters
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
