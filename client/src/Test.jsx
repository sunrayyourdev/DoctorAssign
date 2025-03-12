import { useState } from 'react';

function Chatbot() {
    const [message, setMessage] = useState('');
    const [response, setResponse] = useState('');
    const [patientId, setPatientId] = useState(919); // Replace with actual patientId if dynamic

    const sendMessage = async () => {
        try {
            const res = await fetch("http://127.0.0.1:5010/chatbot_response", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ patientId, content: message }),
            });

            if (!res.ok) {
                throw new Error("Failed to fetch response from chatbot");
            }

            const data = await res.json();
            setResponse(data.response);
        } catch (error) {
            console.error("Error:", error);
            setResponse("Error communicating with chatbot.");
        }
    };

    return (
        <div>
            <h2>Chatbot</h2>
            <input 
                type="text"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Type your message..."
            />
            <button onClick={sendMessage}>Send</button>
            <p><strong>Bot:</strong> {response}</p>
        </div>
    );
}

export default Chatbot;
