import asyncio                                               # run async code from a normal script
from langchain_mcp_adapters.client import MultiServerMCPClient  # connects to one or more MCP servers


async def main():
    # Configure which MCP server(s) to connect to.
    client = MultiServerMCPClient({
        "hybrid_search": {   # server name (must match FastMCP("hybrid_search") in the server file)
            "command": "python",                              # how to launch the server
            "args": [
                r"D:\complete_content_new\llmops-batch\ecomm-prod-assistant\prod_assistant\mcp_servers\product_search_server.py"
            ],  # absolute path to the server script (NOTE: Windows path — change for your Mac)
            "transport": "stdio",                             # talk to server over stdin/stdout
        }
    })

    # Discover tools — ask the server which tools it exposes.
    tools = await client.get_tools()
    print("Available tools:", [t.name for t in tools])

    # Pick tools by name (the two @mcp.tool() functions from the server).
    retriever_tool = next(t for t in tools if t.name == "get_product_info")  # local DB search
    web_tool = next(t for t in tools if t.name == "web_search")              # web fallback

    # --- Step 1: Try retriever first ---
    #query = "Samsung Galaxy S25 price"
    # query = "iPhone 15"
    query = "iPhone 17?"
    retriever_result = await retriever_tool.ainvoke({"query": query})   # call the DB tool
    print("\nRetriever Result:\n", retriever_result)

    # --- Step 2: Fallback to web search if retriever fails ---
    # If DB returned nothing, run a live web search instead.
    if not retriever_result.strip() or "No local results found." in retriever_result:
        print("\n No local results, falling back to web search...\n")
        web_result = await web_tool.ainvoke({"query": query})
        print("Web Search Result:\n", web_result)

if __name__ == "__main__":
    asyncio.run(main())     # start the async main() function
