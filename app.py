from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app)

# Load the model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"  # Replace with your desired model
device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
print("Model loaded.")

# Memory to store conversation history
conversation_memory = []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global conversation_memory

    # Get user input
    user_message = request.json["message"]

    # Append user message to memory
    conversation_memory.append({"role": "user", "content": user_message})

    # Generate response
    context = ""
    for msg in conversation_memory:
        context += f"{msg['role']}: {msg['content']}\n"
    
    # Prepare input IDs for the model
    input_ids = tokenizer.encode(context, return_tensors="pt").to(device)

    # Fix for MPS compatibility (force int32 for attention_mask)
    if device == "mps":
        attention_mask = torch.ones(input_ids.shape, dtype=torch.int32, device=device)
    else:
        attention_mask = torch.ones(input_ids.shape, device=device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=500,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask,  # Use the fixed attention mask
        )
    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Append assistant response to memory
    conversation_memory.append({"role": "assistant", "content": response})

    return jsonify({"response": response})

if __name__ == "__main__":
    socketio.run(app, debug=True)
