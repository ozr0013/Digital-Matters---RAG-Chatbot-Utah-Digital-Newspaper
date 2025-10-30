# ollama_test.py
import ollama

# Pick a small model you have (or pull it first via: ollama pull mistral)
MODEL_NAME = "mistral"   # or "phi3", "llama3", etc.

try:
    print(f"🔹 Sending test prompt to Ollama model: {MODEL_NAME}")
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Hello! Are you connected and working?"}]
    )

    print("\n Ollama response received:\n")
    print(response["message"]["content"])

except Exception as e:
    print(f"\n Error connecting to Ollama: {e}")
    print(" Make sure 'ollama serve' is running in a terminal and the model is pulled.")
