import httpx
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("website browsing")


async def fetch_url_text(url: str, timeout: float = 30.0) -> str:
    """
    Fetches the page text from a URL.

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.
    """
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, timeout=timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            return soup.get_text(separator="\n")
        except httpx.HTTPError:
            return ""

@mcp.tool(description="Fetch text from a website")
async def website_browsing(url: str = None) -> str:
    """
    Fetch the text from a website.
    Args:
        url: The URL to fetch.
    Returns:
        A big chunk of text scraped from the website.
    """
    texts = []

    if url:
        text = await fetch_url_text(url)
        if text:
            print(f"--- From {url} ---\n{text}\n")
            texts.append(f"--- From {url} ---\n{text}\n")

    return "\n".join(texts).strip()

if __name__ == "__main__":
    mcp.run(transport="stdio")
