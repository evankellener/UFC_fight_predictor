import os
from firecrawl import Firecrawl

firecrawl = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))

docs = firecrawl.crawl(url="https://www.ufc.com/events", limit=10)

with open("firecrawl_output.txt", "w", encoding="utf-8") as f:
    for doc in docs:
        if isinstance(doc, dict):
            content = doc.get("content", "")
        else:
            content = str(doc)
        f.write(content)
        f.write("\n\n")

