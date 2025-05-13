from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os
import httpx
import logging

load_dotenv()

mcp = FastMCP("internet search")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not SERPAPI_API_KEY:
    logging.error("SERPAPI_API_KEY environment variable is not set!")
    raise ValueError("SERPAPI_API_KEY environment variable is required")
else:
    logging.info(f"API Key found (starts with): {SERPAPI_API_KEY}...")

SERPER_URL = "https://google.serper.dev/search"

@mcp.tool(description="Search the web using Serper.dev")
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
    payload = {"q": q }

    headers = {
        "X-API-KEY": SERPAPI_API_KEY,
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:
        try:
            logging.info(f"Making request to Serper.dev with query: {q}")
            resp = await client.post(SERPER_URL, headers=headers, json=payload, timeout=30.0)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            if e.response.status_code == 403:
                logging.error("403 Forbidden - Please check your SERPAPI_API_KEY")
            return {"organic": []}
        except httpx.RequestError as e:
            logging.error(f"Request error occurred: {str(e)}")
            return {"organic": []}
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            return {"organic": []}

if __name__ == "__main__":
    mcp.run(transport="stdio")
