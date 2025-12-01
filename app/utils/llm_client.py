# app/llm_client.py

from openai import OpenAI
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()

def run_llm(prompt):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return "❌ ERROR: OPENAI_API_KEY not found. Make sure it is in your .env file."

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content
