from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os
import httpx

load_dotenv()

mcp = FastMCP("internet search")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
SERPER_URL = "https://google.serper.dev/search"

@mcp.tool()
async def search_web(query: str, num: int = 3, site: str | None = None) -> dict:
    """
    Uses Serper.dev to search the web.

    Args:
        query: The free-text search query.
        num: How many top results to return.
        site: If provided, restricts the search to a specific domain (e.g. "docs.python.org").

    Returns:
        JSON response from Serper (contains 'organic' list of results).
    """
    q = f"{'site:' + site if site else ''} {query}".strip()
    payload = {"q": q, "num": num}

    headers = {
        "X-API-KEY": SERPAPI_API_KEY,
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(SERPER_URL, headers=headers, json=payload, timeout=30.0)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError:
            return {"organic": []}

if __name__ == "__main__":
    mcp.run(transport="stdio")
