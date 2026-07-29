# ── Imports ───────────────────────────────────────────────────────────────
from mcp.server.fastmcp import FastMCP                        # MCP server framework (exposes tools to an AI client)
from retriever.retrieval import Retriever                     # our vector-DB retriever (fetches product reviews)
from langchain_community.tools import DuckDuckGoSearchRun     # web-search tool (fallback when DB has nothing)

# Initialize MCP server. "hybrid_search" = the server's name the client connects to.
mcp = FastMCP("hybrid_search")

# Load retriever once (at startup) so every request reuses the same connection — not rebuilt per call.
retriever_obj = Retriever()
retriever = retriever_obj.load_retriever()

# LangChain DuckDuckGo tool — used for live web searches.
duckduckgo = DuckDuckGoSearchRun()

# ---------- Helpers ----------
def format_docs(docs) -> str:
    """Format retriever docs into readable context."""
    if not docs:                              # nothing retrieved -> empty string
        return ""
    formatted_chunks = []
    for d in docs:                            # for each retrieved review doc...
        meta = d.metadata or {}               # product info lives in metadata
        formatted = (
            f"Title: {meta.get('product_title', 'N/A')}\n"   # product title
            f"Price: {meta.get('price', 'N/A')}\n"           # price
            f"Rating: {meta.get('rating', 'N/A')}\n"         # rating
            f"Reviews:\n{d.page_content.strip()}"            # the review text
        )
        formatted_chunks.append(formatted)
    return "\n\n---\n\n".join(formatted_chunks)   # join docs with a separator

# ---------- MCP Tools ----------
# @mcp.tool() = expose this function to the AI client as a callable tool.
@mcp.tool()
async def get_product_info(query: str) -> str:
    """Retrieve product information for a given query from local retriever."""
    try:
        docs = retriever.invoke(query)        # search the vector DB
        context = format_docs(docs)           # turn results into readable text
        if not context.strip():               # empty -> tell client no local hit
            return "No local results found."
        return context
    except Exception as e:
        return f"Error retrieving product info: {str(e)}"   # never crash the server; return error text

@mcp.tool()
async def web_search(query: str) -> str:
    """Search the web using DuckDuckGo if retriever has no results."""
    try:
        return duckduckgo.run(query)          # run a live web search
    except Exception as e:
        return f"Error during web search: {str(e)}"

# ---------- Run Server ----------
if __name__ == "__main__":
    #mcp.run(transport="stdio")               # stdio transport (client launches server as subprocess)
    mcp.run(transport="streamable-http")      # HTTP transport (client connects over HTTP) — active one
