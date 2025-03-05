import { useState } from 'react';

function Chatbot() {
    const [message, setMessage] = useState('');
    const [response, setResponse] = useState('');

    const sendMessage = async () => {
        const res = await fetch("http://localhost:5000/chatbot_response", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message }),
        });

        const data = await res.json();
        setResponse(data.response);
    };

    return (
        <div>
            <input value={message} onChange={(e) => setMessage(e.target.value)} />
            <button onClick={sendMessage}>Send</button>
            <p>Bot: {response}</p>
        </div>
    );
}

export default Chatbot;
