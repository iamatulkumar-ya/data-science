from fastmcp.server.lifespan import lifespan
from fastmcp import FastMCP
from starlette.routing import Route

# Lifespans let our run code once when the server starts and clean up when it stops. Unlike per-session handlers, 
# lifespans run exactly once regardless of how many clients connect.
@lifespan
async def mcp_lifespan(server):

    try:
        print("ServiceNow MCP is spinning up...")
        yield

    finally:
        print("ServiceNow MCP is shutting down...")


mcp = FastMCP("servicenow_mcp_server", lifespan=mcp_lifespan)

#region REGISTER TOOLS ********


#endregion


#region REGISTER HTTP ENDPOINTS ********

app = mcp.http_app(path="api")

def do_processing():
    pass

app.router.routes.extend([
    Route("/do_process", endpoint=do_processing, methods=["GET"])
])

#endregion


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)