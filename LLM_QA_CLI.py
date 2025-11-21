#!/usr/bin/env python3
"""
LLM_QA_CLI.py
Simple CLI that preprocesses a question, constructs a prompt, and sends it to an LLM API.
It uses the OpenAI API if OPENAI_API_KEY is set in the environment. Otherwise it returns a mock response.
"""

import os
import re
import sys
import json
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

def preprocess(text):
    # basic preprocessing: lowercase, remove punctuation (keep spaces), tokenize by whitespace
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    tokens = text.split()
    return ' '.join(tokens), tokens

def build_prompt(processed_question):
    prompt = f"""You are a helpful assistant. Answer the question concisely.
Question (processed): {processed_question}

Provide:
1) A short direct answer.
2) A one-sentence explanation of the answer.
Respond in JSON with keys: answer, explanation.
"""
    return prompt

def call_openai(prompt):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    client = OpenAI(api_key=api_key)
    # Using the Chat Completions style via the OpenAI python client (kept minimal)
    resp = client.responses.create(model="gpt-4o-mini", input=prompt, max_tokens=300)
    # try to parse JSON from the text response
    text = resp.output_text if hasattr(resp, "output_text") else str(resp)
    return text

def parse_json_like(text):
    try:
        # attempt to find a JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end]
            return json.loads(candidate)
    except Exception:
        pass
    # fallback: return as plain text
    return {"answer": text.strip(), "explanation": ""}

def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter your question: ").strip()
    processed, tokens = preprocess(question)
    prompt = build_prompt(processed)
    print("="*40)
    print("Processed question:", processed)
    print("Tokens:", tokens)
    print("- Sending prompt to LLM (local or API) -")
    api_text = call_openai(prompt)
    if api_text is None:
        # mock response (useful if API key not set)
        mock = {
            "answer": "This is a mocked answer because no API key was found.",
            "explanation": "Set OPENAI_API_KEY in your environment to use a real LLM."
        }
        result = mock
    else:
        result = parse_json_like(api_text)
    print("\nLLM Answer:\n", result.get("answer"))
    print("\nExplanation:\n", result.get("explanation"))

if __name__ == "__main__":
    main()
