import os
from openai import OpenAI

key = os.getenv("DEEPSEEK_API_KEY")
if not key:
    raise RuntimeError("DEEPSEEK_API_KEY not set")

client = OpenAI(api_key=key, base_url="https://api.deepseek.com/v1")

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Reply with: OK"}],
    temperature=0.0,
    max_tokens=10,
)

print(resp.choices[0].message.content)
