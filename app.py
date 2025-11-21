from flask import Flask, render_template, request, jsonify
import os, re, json
app = Flask(__name__)

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return ' '.join(tokens), tokens

def build_prompt(processed_question):
    prompt = f"You are a helpful assistant. Answer the question concisely. Question: {processed_question}"
    return prompt

def call_openai(prompt):
    # keep this optional — returns a mock if API key not set
    try:
        from openai import OpenAI
    except Exception:
        OpenAI = None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    client = OpenAI(api_key=api_key)
    resp = client.responses.create(model="gpt-4o-mini", input=prompt, max_tokens=300)
    text = resp.output_text if hasattr(resp, "output_text") else str(resp)
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        question = request.form.get("question", "")
        processed, tokens = preprocess(question)
        prompt = build_prompt(processed)
        api_text = call_openai(prompt)
        if api_text is None:
            response = {
                "answer": "Mock answer (no API key).",
                "explanation": "Set OPENAI_API_KEY in environment to enable real LLM responses."
            }
        else:
            # simple attempt to extract JSON from model output
            start = api_text.find("{")
            end = api_text.rfind("}")+1
            if start!=-1 and end!=-1 and end>start:
                try:
                    response = json.loads(api_text[start:end])
                except Exception:
                    response = {"answer": api_text, "explanation": ""}
            else:
                response = {"answer": api_text, "explanation": ""}
        return render_template("index.html", question=question, processed=processed, tokens=tokens, response=response, prompt=prompt)
    return render_template("index.html", question="", processed="", tokens=[], response=None, prompt="")

if __name__ == "__main__":
    # For local testing only. When deploying, use a WSGI server as recommended by your host.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
