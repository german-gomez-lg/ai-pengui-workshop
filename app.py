from __future__ import annotations

import logging
import uvicorn
import os

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

from src.agentiv.websocket_entry import mount_agui_websocket, AgentTelemetryProtocol
from src.agentiv.state_store import build_agentiv_state_store_from_env

logger = logging.getLogger("MAIN_APP")

## =========================
## Configuration Zone
# Fast API Application Name
APPLICATION_NAME = "Echo Agent"

# Flow name of the agent
PENGUIFLOW_FLOW_NAME = "echo_agent"

# Change the package name to the correct one, eg: "echo_agent"
from src.echo_agent.config import Config
# Change the class name to the correct Orchestrator class name: Eg: EchoAgentOrchestrator
from src.echo_agent.orchestrator import EchoAgentOrchestrator as Orchestrator
## End of Configuration Zone
##==========================

def create_app() -> FastAPI:
    # Load configuration from environment variables
    config = Config.from_env()

    # Create fast api application
    app = FastAPI(title=APPLICATION_NAME)

    # Instantiate state store
    state_store = build_agentiv_state_store_from_env()
    app.state.config = config

    # Define factory function for orchestrator
    def orchestrator_factory(telemetry: AgentTelemetryProtocol) -> Orchestrator:
        return Orchestrator(config=config, telemetry=telemetry) 
    
    # Mount websocket endpoint to our app
    mount_agui_websocket(app=app,
                         state_store=state_store,
                         flow_name=PENGUIFLOW_FLOW_NAME,
                         orchestrator_factory=orchestrator_factory)
    return app

# Create app
app = create_app()

# Configure redirect for standard swagger documentation
@app.get('/')
def root(request: Request):
    """Just redirect to the docs"""
    return RedirectResponse(url='/docs')

# Define entry point
def main():
    port = int(os.environ.get("DATABRICKS_APP_PORT", 6969))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False
    )

# Application entry point
if __name__ == "__main__":
    main()