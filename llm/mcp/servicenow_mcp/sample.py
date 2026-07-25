import logging
from contextlib import asynccontextmanager
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from fastmcp import FastMCP

# 1. Setup logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent_system")

# 2. Initialize FastMCP for your AI tools
mcp = FastMCP("Agent Operations Framework")

# ---------------------------------------------------------
# AI AGENT TOOLS (Exposed to Claude)
# ---------------------------------------------------------
@mcp.tool()
async def process_ticket(ticket_id: str, action: str) -> str:
    """Execute automated actions on a system ticket."""
    logger.info(f"Agent executing action '{action}' on ticket {ticket_id}")
    return f"Ticket {ticket_id} successfully updated with action: {action}"


# ---------------------------------------------------------
# HTTP REST ENDPOINTS (Exposed for Administration)
# ---------------------------------------------------------
async def get_logs(request):
    """HTTP GET endpoint to fetch recent system activity logs."""
    # Example logic to return real-time system state
    return JSONResponse({
        "status": "success",
        "recent_events": ["Agent connected", "Executed process_ticket"]
    })

async def close_session(request):
    """HTTP POST endpoint to clear session states or disconnect an agent."""
    try:
        body = await request.json()
        session_id = body.get("session_id")
        logger.info(f"Administrative override: Closing session {session_id}")
        return JSONResponse({"status": "terminated", "session_id": session_id})
    except Exception as e:
        return JSONResponse({"error": "Invalid payload", "details": str(e)}, status_code=400)


# ---------------------------------------------------------
# THE COUPLING (FastMCP Engine + Starlette Routing)
# ---------------------------------------------------------
# Extract the base Starlette application instance
app = mcp.asgi()

# Explicitly mount your administrative REST paths onto the Starlette app
app.router.routes.extend([
    Route("/admin/logs", endpoint=get_logs, methods=["GET"]),
    Route("/admin/session/close", endpoint=close_session, methods=["POST"]),
])

# If you need to run this as a standalone server over network/SSE:
if __name__ == "__main__":
    import uvicorn
    # This runs the web server hosting both your MCP endpoints and your admin routes
    uvicorn.run(app, host="0.0.0.0", port=8000)