
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const chatMessages = document.getElementById("chat-messages");
const newChatBtn = document.getElementById("new-chat-btn");
const sendBtn = document.getElementById("send-btn");

function addMessage(text, sender) {
    const message = document.createElement("div");
    message.className = `message ${sender}`;
    message.textContent = text;

    message.style.padding = "14px 18px";
    message.style.borderRadius = "12px";
    message.style.maxWidth = "80%";
    message.style.whiteSpace = "pre-wrap";
    message.style.overflowWrap = "anywhere";
    message.style.alignSelf =
        sender === "user" ? "flex-end" : "flex-start";
    message.style.background =
        sender === "user" ? "#303030" : "#262626";

    chatMessages.appendChild(message);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return message;
}

chatForm.addEventListener("submit", async function (event) {
    event.preventDefault();

    const question = userInput.value.trim();
    if (!question) return;

    const welcome = chatMessages.querySelector(".welcome-message");
    if (welcome) welcome.remove();

    addMessage(question, "user");

    userInput.value = "";
    userInput.disabled = true;
    sendBtn.disabled = true;

    const loadingMessage = addMessage("Thinking...", "assistant");

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 45000);

    try {
        const response = await fetch("http://127.0.0.1:8000/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question }),
            signal: controller.signal
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || `HTTP error: ${response.status}`);
        }

        // Render the answer as Markdown when Marked.js is available.
        if (typeof marked !== "undefined") {
            loadingMessage.innerHTML = marked.parse(data.answer);
        } else {
            loadingMessage.textContent = data.answer;
        }

    } catch (error) {
        console.error("Chat error:", error);

        if (error.name === "AbortError") {
            loadingMessage.textContent =
                "The request took too long. Please try again.";
        } else if (error.message.startsWith("HTTP error:") ||
                   error.message.includes("Gemini")) {
            loadingMessage.textContent = error.message;
        } else {
            loadingMessage.textContent =
                "Could not connect to the backend. Check that FastAPI is running.";
        }

    } finally {
        clearTimeout(timeoutId);
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
    }
});

newChatBtn.addEventListener("click", function () {
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <h1>How can I help you?</h1>
            <p>Ask me anything about your uploaded knowledge base.</p>
        </div>
    `;

    userInput.value = "";
    userInput.disabled = false;
    sendBtn.disabled = false;
    userInput.focus();
});

userInput.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        chatForm.requestSubmit();
    }
});
