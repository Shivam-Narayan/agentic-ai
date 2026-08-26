"""Quick test to verify Tavily API key is working."""

import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("TAVILY_API_KEY")
if not key:
    print("❌ TAVILY_API_KEY not found in .env")
    exit(1)

print(f"✅ Key found: {key[:8]}...")

try:
    from langchain_community.tools.tavily_search import TavilySearchResults

    tool = TavilySearchResults(k=2)
    results = tool.invoke({"query": "current weather in Bangalore India"})

    print("\n✅ Tavily is working! Results:\n")
    for r in results:
        print(f"  URL    : {r.get('url', 'N/A')}")
        print(f"  Snippet: {r.get('content', '')[:150]}...")
        print()

except Exception as e:
    print(f"\n❌ Tavily call failed: {e}")
