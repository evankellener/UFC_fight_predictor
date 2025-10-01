from firecrawl import Firecrawl

firecrawl = Firecrawl(api_key="fc-2497518e738344388a2c77da41c64fdd")

docs = firecrawl.crawl(url="https://mmajunkie.usatoday.com/story/sports/ufc/2025/09/26/ufc-perth-dominick-reyes-vs-carlos-ulberg-weigh-in-results/86366809007/", limit=10)

with open("firecrawl_output.txt", "w", encoding="utf-8") as f:
    for doc in docs:
        if isinstance(doc, dict):
            content = doc.get("content", "")
        else:
            content = str(doc)
        f.write(content)
        f.write("\n\n")

