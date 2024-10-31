from flask import Flask, request, jsonify, render_template
from datetime import datetime
import subprocess
import json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def query_llama_claude(prompt):
    # Run the ollama CLI command to get a response from the llama-claude model
    result = subprocess.run(
        ["ollama", "run", "llama-claude"],
        input=prompt,          # Removed .encode() since input should already be a string
        capture_output=True,    # Capture the output
        text=True
    )
    # Process the model's output
    return result.stdout.strip()

@app.route('/ask', methods=['POST'])
def ask_petgpt():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        question = data.get('question')
        if not question:
            return jsonify({"error": "No question provided"}), 400

        # Generate response using the LlamaClaude model
        answer = query_llama_claude(question)

        return jsonify({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
