# Chatbot-assistant
create a chatbot assist for online purchase using Gpt-neo -1.3b
Development Plan
1. Prerequisites
Install Python Packages:
!pip install transformers flask flask-socketio torch

transformers for the LLM.
flask for creating the web server.
flask-socketio for enabling real-time communication.
torch for running LLMs on your Mac M1 (optimized with torch's mps backend).
Set Up Local LLM:

Use an open-source model (e.g., Llama 2 or GPT-NeoX) from Hugging Face.
2. Memory Mechanism
Store conversation history as a list of dictionaries, maintaining the user and assistant messages.
Append each message to the memory list to enable context continuity.
3. Create the Chat Interface
Use HTML and JavaScript to create a front-end for users to type messages and see responses.
Use Flask to serve the interface and handle the backend.
4. Run the Chatbot Locally
Run the Flask app, access the chatbot via a browser, and interact with it.
