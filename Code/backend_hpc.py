from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

HF_TOKEN = os.getenv("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")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set.")

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

print("Loading model... This may take a while on first run.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=HF_TOKEN
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model loaded successfully.")

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    prompt = f"<|system|>\nYou are a helpful assistant specialised in formal verification, safety/security, and mathematical proofs.\n<|user|>\n{user_input}\n<|assistant|>\n"

    outputs = pipe(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)
    reply = outputs[0]["generated_text"].replace(prompt, "").strip()

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)  
