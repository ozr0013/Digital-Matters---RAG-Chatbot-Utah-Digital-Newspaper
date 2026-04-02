/**
 * Utah Digital Newspapers Archive - Frontend JavaScript
 */

const API_BASE_URL = '/api';
const CHAT_MESSAGES_CONTAINER = document.getElementById('chatMessages');
const MESSAGE_INPUT = document.getElementById('messageInput');
const SEND_BUTTON = document.getElementById('sendButton');
const MENU_BUTTON = document.getElementById('menuButton');
const HOME_BUTTON = document.getElementById('homeButton');
const SIDEBAR = document.getElementById('sidebar');
const SIDEBAR_OVERLAY = document.getElementById('sidebarOverlay');

// State
let isLoading = false;
let chatHistory = [];       // messages in the current session
let sessions = [];          // all saved sessions
let currentSessionId = null;

// ======================
// Initialization
// ======================

document.addEventListener('DOMContentLoaded', () => {
    SEND_BUTTON.addEventListener('click', sendMessage);
    MESSAGE_INPUT.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !isLoading) sendMessage();
    });
    MENU_BUTTON.addEventListener('click', openSidebar);
    HOME_BUTTON.addEventListener('click', goHome);

    // Load saved sessions into sidebar
    loadSessions();

    // Restore last active session (if any)
    try {
        const current = localStorage.getItem('currentSession');
        if (current) {
            const parsed = JSON.parse(current);
            currentSessionId = parsed.id;
            chatHistory = parsed.messages || [];
            if (chatHistory.length > 0) {
                removeWelcomeScreen();
                chatHistory.forEach(renderMessageFromHistory);
            }
        }
    } catch (e) {
        console.warn('Could not restore session:', e);
    }

    // Migrate old flat chatHistory format
    try {
        const old = localStorage.getItem('chatHistory');
        if (old && !localStorage.getItem('currentSession')) {
            const msgs = JSON.parse(old);
            if (msgs.length > 0) {
                chatHistory = msgs;
                removeWelcomeScreen();
                chatHistory.forEach(renderMessageFromHistory);
            }
            localStorage.removeItem('chatHistory');
        }
    } catch (e) {}

    renderSidebar();
});

// ======================
// Home / Session Nav
// ======================

function goHome() {
    saveCurrentSession();
    currentSessionId = null;
    chatHistory = [];
    localStorage.removeItem('currentSession');
    showWelcomeScreen();
    renderSidebar();
}

function showWelcomeScreen() {
    CHAT_MESSAGES_CONTAINER.innerHTML = `
        <div class="welcome-container" id="welcomeContainer">
            <h2>Discover Utah's Past</h2>
            <p class="welcome-description">
                Search through historical newspaper articles from across Utah.
                This archive contains <em>thousands of digitized pages</em> spanning
                over a century of local journalism, community stories, and historical events.
            </p>
            <div class="example-queries">
                <p class="examples-title">Try searching for</p>
                <div class="example-links">
                    <button class="example-link" onclick="setQuery('What articles mention women\\'s suffrage in Utah?')">
                        Women's suffrage movement in Utah
                    </button>
                    <button class="example-link" onclick="setQuery('Find articles about the 1918 Spanish flu in Salt Lake City')">
                        The 1918 Spanish flu in Salt Lake City
                    </button>
                    <button class="example-link" onclick="setQuery('Show me mining related articles from the 1900s')">
                        Early 1900s mining industry coverage
                    </button>
                    <button class="example-link" onclick="setQuery('What newspapers covered the first transcontinental railroad?')">
                        Transcontinental railroad completion
                    </button>
                </div>
            </div>

            <p class="built-by">
                RAG system developed by <strong>Omar Rizwan</strong> &mdash; Software Engineer &amp; University of Utah Student
            </p>
        </div>
    `;
}

function removeWelcomeScreen() {
    const w = document.querySelector('.welcome-container');
    if (w) w.remove();
}

// ======================
// Session Management
// ======================

function loadSessions() {
    try {
        const saved = localStorage.getItem('sessions');
        sessions = saved ? JSON.parse(saved) : [];
    } catch (e) {
        sessions = [];
    }
}

function saveCurrentSession() {
    if (chatHistory.length === 0) return;

    const firstQuery = chatHistory.find(m => m.role === 'user');
    const title = firstQuery ? firstQuery.message : 'Untitled';
    const dateStr = new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });

    if (currentSessionId) {
        const idx = sessions.findIndex(s => s.id === currentSessionId);
        if (idx >= 0) {
            sessions[idx].messages = [...chatHistory];
            sessions[idx].title = title.substring(0, 70);
        } else {
            sessions.unshift({
                id: currentSessionId,
                title: title.substring(0, 70),
                date: dateStr,
                messages: [...chatHistory]
            });
        }
    } else {
        currentSessionId = Date.now().toString();
        sessions.unshift({
            id: currentSessionId,
            title: title.substring(0, 70),
            date: dateStr,
            messages: [...chatHistory]
        });
    }

    if (sessions.length > 30) sessions = sessions.slice(0, 30);

    localStorage.setItem('sessions', JSON.stringify(sessions));
    localStorage.setItem('currentSession', JSON.stringify({ id: currentSessionId, messages: chatHistory }));
}

function loadSession(id) {
    saveCurrentSession();

    const session = sessions.find(s => s.id === id);
    if (!session) return;

    currentSessionId = id;
    chatHistory = [...session.messages];
    localStorage.setItem('currentSession', JSON.stringify({ id, messages: chatHistory }));

    CHAT_MESSAGES_CONTAINER.innerHTML = '';
    chatHistory.forEach(renderMessageFromHistory);
    CHAT_MESSAGES_CONTAINER.scrollTop = CHAT_MESSAGES_CONTAINER.scrollHeight;

    renderSidebar();
    closeSidebar();
}

function deleteSession(id, event) {
    event.stopPropagation();
    sessions = sessions.filter(s => s.id !== id);
    localStorage.setItem('sessions', JSON.stringify(sessions));

    if (currentSessionId === id) {
        currentSessionId = null;
        chatHistory = [];
        localStorage.removeItem('currentSession');
        showWelcomeScreen();
    }

    renderSidebar();
}

function clearAllSessions() {
    sessions = [];
    currentSessionId = null;
    chatHistory = [];
    localStorage.removeItem('sessions');
    localStorage.removeItem('currentSession');
    localStorage.removeItem('chatHistory');
    showWelcomeScreen();
    renderSidebar();
    closeSidebar();
}

function renderSidebar() {
    const list = document.getElementById('sessionsList');
    if (!list) return;

    if (sessions.length === 0) {
        list.innerHTML = '<p class="no-sessions">No previous searches yet</p>';
        return;
    }

    list.innerHTML = sessions.map(s => `
        <div class="session-item ${s.id === currentSessionId ? 'session-active' : ''}"
             onclick="loadSession('${s.id}')">
            <div class="session-info">
                <span class="session-title">${escapeHtml(s.title)}</span>
                <span class="session-date">${escapeHtml(s.date || '')}</span>
            </div>
            <button class="session-delete" onclick="deleteSession('${s.id}', event)" title="Remove">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `).join('');
}

// ======================
// Sending Messages
// ======================

async function sendMessage() {
    const message = MESSAGE_INPUT.value.trim();
    if (!message || isLoading) return;

    isLoading = true;
    SEND_BUTTON.disabled = true;
    MESSAGE_INPUT.disabled = true;

    removeWelcomeScreen();
    addMessageToChat('user', message);
    MESSAGE_INPUT.value = '';

    chatHistory.push({ role: 'user', message });
    saveCurrentSession();
    renderSidebar();

    // Loading indicator
    const loadingId = 'loading-' + Date.now();
    const loadingDiv = document.createElement('div');
    loadingDiv.id = loadingId;
    loadingDiv.className = 'message assistant-message loading';
    loadingDiv.innerHTML = '<div class="message-content"><p>Searching the archive...</p></div>';
    CHAT_MESSAGES_CONTAINER.appendChild(loadingDiv);
    CHAT_MESSAGES_CONTAINER.scrollTop = CHAT_MESSAGES_CONTAINER.scrollHeight;

    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });

        const loader = document.getElementById(loadingId);
        if (loader) loader.remove();

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const data = await response.json();

        if (data.error) {
            addMessageToChat('assistant', `<p>Error: ${escapeHtml(data.error)}</p>`);
        } else {
            addMessageToChat('assistant', formatAssistantResponse(data));
            chatHistory.push({ role: 'assistant', message: data.answer, sources: data.sources });
            saveCurrentSession();
            renderSidebar();
        }
    } catch (error) {
        const loader = document.getElementById(loadingId);
        if (loader) loader.remove();
        console.error('Error:', error);
        addMessageToChat('assistant',
            `<p>I encountered an error while processing your request. Please try again. (${escapeHtml(error.message)})</p>`
        );
    } finally {
        isLoading = false;
        SEND_BUTTON.disabled = false;
        MESSAGE_INPUT.disabled = false;
        MESSAGE_INPUT.focus();
    }
}

// ======================
// Rendering
// ======================

function renderMessageFromHistory(item) {
    if (item.role === 'user') {
        addMessageToChat('user', item.message);
    } else {
        addMessageToChat('assistant', formatAssistantResponse({
            answer: item.message,
            sources: item.sources || []
        }));
    }
}

function addMessageToChat(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    if (role === 'assistant') {
        contentDiv.innerHTML = content;
    } else {
        contentDiv.innerHTML = `<p>${escapeHtml(content)}</p>`;
    }

    messageDiv.appendChild(contentDiv);
    CHAT_MESSAGES_CONTAINER.appendChild(messageDiv);
    CHAT_MESSAGES_CONTAINER.scrollTop = CHAT_MESSAGES_CONTAINER.scrollHeight;
}

function formatAssistantResponse(data) {
    const answerHtml = escapeHtml(data.answer).replace(/\n/g, '<br>');
    let html = `<p>${answerHtml}</p>`;

    if (data.sources && data.sources.length > 0) {
        html += '<div class="sources-section">';
        html += '<h4>Sources</h4>';
        html += '<div class="sources-list">';

        data.sources.forEach((source, index) => {
            let metaLine = '';
            if (source.paper) metaLine += source.paper;
            if (source.date) metaLine += metaLine ? ` · ${source.date}` : source.date;
            if (source.relevance) metaLine += metaLine ? ` · ${source.relevance} match` : `${source.relevance} match`;

            const linkHtml = source.link
                ? `<a href="${escapeHtml(source.link)}" target="_blank" rel="noopener noreferrer" class="source-link">
                     <i class="fas fa-external-link-alt"></i> View Original
                   </a>`
                : '';

            html += `
                <div class="source-item">
                    <div class="source-number">${index + 1}</div>
                    <div class="source-content">
                        <h5>${escapeHtml(source.title || 'Untitled Article')}</h5>
                        ${metaLine ? `<p class="source-meta">${escapeHtml(metaLine)}</p>` : ''}
                        <p>${escapeHtml(source.snippet || '')}</p>
                        ${linkHtml}
                    </div>
                </div>
            `;
        });

        html += '</div></div>';
    }

    return html;
}

// ======================
// Sidebar
// ======================

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

// ======================
// Utilities
// ======================

function setQuery(query) {
    MESSAGE_INPUT.value = query;
    MESSAGE_INPUT.focus();
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
