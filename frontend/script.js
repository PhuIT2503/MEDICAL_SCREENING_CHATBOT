const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');

function appendMessage(text, sender) {
    const div = document.createElement('div');
    div.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');
    div.textContent = text;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight; // Tự cuộn xuống dưới
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    // 1. Hiện tin nhắn người dùng
    appendMessage(text, 'user');
    userInput.value = '';

    // 2. Hiện trạng thái đang nhập...
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'typing-indicator';
    loadingDiv.textContent = 'Bác sĩ AI đang suy nghĩ...';
    chatBox.appendChild(loadingDiv);

    try {
        // 3. Gọi API (Kết nối với Python Backend)
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: text })
        });

        const data = await response.json();
        
        // 4. Xóa loading và hiện câu trả lời
        chatBox.removeChild(loadingDiv);
        appendMessage(data.answer, 'bot');

    } catch (error) {
        chatBox.removeChild(loadingDiv);
        appendMessage("Lỗi kết nối server. Vui lòng thử lại!", 'bot');
        console.error('Error:', error);
    }
}

function handleEnter(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}