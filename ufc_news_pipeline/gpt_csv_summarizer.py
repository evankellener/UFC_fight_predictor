import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Read the content from 'firecrawl_output.txt'
with open("firecrawl_output.txt", "r", encoding="utf-8") as f:
    content = f.read()

question = (
    "Did anyone withdraw from the fight? If so, who withdrew, and who took their place? Was there a short notice replacement?"
)

# Replace with your OpenAI API key or ensure it's set in your environment
 # <-- Replace with your OpenAI API key

response = client.chat.completions.create(model="gpt-3.5-turbo",
messages=[
    {"role": "system", "content": "You are a helpful assistant that answers questions about UFC fight news articles."},
    {"role": "user", "content": f"{content}\n\n{question}"},
],
max_tokens=256,
temperature=0.2)

print("OpenAI response:")
print(response.choices[0].message.content)