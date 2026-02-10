"""Databricks tool to check campaign line status by line ID."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from penguiflow.catalog import tool
from penguiflow.planner import ToolContext


class CampaignLineStatusArgs(BaseModel):
    """Inputs for checking a campaign line."""

    line_id: int = Field(description="Line item ID to look up")
    warehouse_id: str | None = Field(
        default=None,
        description="Optional Databricks SQL warehouse ID (overrides tool context)",
    )


class CampaignLineStatusResult(BaseModel):
    """Output for a campaign line status check."""

    found: bool
    line_id: int
    line_item_status: str | None = None
    campaign_id: int | None = None
    campaign_status: str | None = None
    line_item_name: str | None = None
    campaign_name: str | None = None
    message: str


def _get_workspace_client(ctx: ToolContext | None) -> Any:
    if ctx is None:
        raise RuntimeError("Missing tool context for Databricks authentication")
    tool_ctx = ctx.tool_context if hasattr(ctx, "tool_context") else None
    if not isinstance(tool_ctx, dict):
        raise RuntimeError("Missing tool context for Databricks authentication")
    return tool_ctx.get("databricks_workspace_client")


def _get_warehouse_id(ctx: ToolContext | None) -> str | None:
    if ctx is None:
        return None
    tool_ctx = ctx.tool_context if hasattr(ctx, "tool_context") else None
    if isinstance(tool_ctx, dict):
        value = tool_ctx.get("databricks_warehouse_id")
        return str(value) if value else None
    return None


def _select_warehouse_id(workspace_client: Any) -> str:
    warehouses = list(workspace_client.warehouses.list())
    if not warehouses:
        raise RuntimeError("No SQL warehouses available in Databricks workspace")
    running = [w for w in warehouses if str(getattr(w, "state", "")).upper() == "RUNNING"]
    selected = running[0] if running else warehouses[0]
    return str(selected.id)


def _execute_query(workspace_client: Any, warehouse_id: str, sql: str) -> tuple[list[dict[str, Any]], list[str]]:
    if hasattr(workspace_client, "statement_execution"):
        response = workspace_client.statement_execution.execute_statement(
            statement=sql,
            warehouse_id=warehouse_id,
        )
        result = response.result
        manifest = response.manifest
        cols = (
            [c.name for c in manifest.schema.columns]
            if manifest and manifest.schema and manifest.schema.columns
            else []
        )
        rows: list[dict[str, Any]] = []
        if result and result.data_array:
            for arr in result.data_array:
                row = {}
                for i, v in enumerate(arr):
                    key = cols[i] if i < len(cols) else f"col_{i}"
                    row[key] = v
                rows.append(row)
        return rows, cols

    if hasattr(workspace_client, "sql") and hasattr(workspace_client.sql, "statements"):
        response = workspace_client.sql.statements.execute(
            statement=sql,
            warehouse_id=warehouse_id,
        )
        result = workspace_client.sql.statements.get_result(response.statement_id)
        cols = [c.name for c in getattr(result, "columns", [])]
        rows: list[dict[str, Any]] = []
        for arr in getattr(result, "data_array", []) or []:
            row = {}
            for i, v in enumerate(arr):
                key = cols[i] if i < len(cols) else f"col_{i}"
                row[key] = v
            rows.append(row)
        return rows, cols

    raise RuntimeError("Databricks SQL statement execution API not available")


@tool(desc="Check a campaign line status in Databricks", side_effects="read", tags=["databricks"])
async def check_campaign_line_status(
    args: CampaignLineStatusArgs,
    ctx: ToolContext,
) -> CampaignLineStatusResult:
    if args.line_id <= 0:
        return CampaignLineStatusResult(
            found=False,
            line_id=args.line_id,
            message="Line ID must be a positive integer.",
        )

    try:
        from databricks.sdk import WorkspaceClient
    except Exception as exc:  # pragma: no cover - environment dependent
        return CampaignLineStatusResult(
            found=False,
            line_id=args.line_id,
            message=f"Databricks SDK not available: {exc}",
        )

    workspace_client = _get_workspace_client(ctx) or WorkspaceClient()
    warehouse_id = args.warehouse_id or _get_warehouse_id(ctx) or _select_warehouse_id(workspace_client)

    sql = f"""
    select
        li.id as line_item_id,
        li.name as line_item_name,
        li.status as line_item_status,
        c.cid as campaign_id,
        c.campaign_name as campaign_name,
        c.status as campaign_status
    from us_data_science.data_container.v_dim_mosaic_line_item_raw li
    left join us_data_science.data_container.v_dim_mosaic_campaign_raw c
        on li.cid = c.cid
    where li.id = {args.line_id}
    """

    try:
        rows, cols = _execute_query(workspace_client, warehouse_id, sql)
    except Exception as exc:
        return CampaignLineStatusResult(
            found=False,
            line_id=args.line_id,
            message=f"Query failed for line {args.line_id}: {exc}",
        )

    if len(rows) != 1:
        if len(rows) == 0:
            return CampaignLineStatusResult(
                found=False,
                line_id=args.line_id,
                message=f"No line item found for ID {args.line_id}.",
            )
        return CampaignLineStatusResult(
            found=False,
            line_id=args.line_id,
            message=f"Multiple rows returned for line ID {args.line_id}.",
        )

    row = rows[0]
    line_item_status = row.get("line_item_status")
    campaign_status = row.get("campaign_status")
    line_item_name = row.get("line_item_name")
    campaign_id = row.get("campaign_id")
    campaign_name = row.get("campaign_name")

    message = (
        f"Line {args.line_id} status: {line_item_status}. "
        f"Campaign {campaign_id} status: {campaign_status}."
    )

    return CampaignLineStatusResult(
        found=True,
        line_id=args.line_id,
        line_item_status=str(line_item_status) if line_item_status is not None else None,
        campaign_id=int(campaign_id) if campaign_id is not None else None,
        campaign_status=str(campaign_status) if campaign_status is not None else None,
        line_item_name=str(line_item_name) if line_item_name is not None else None,
        campaign_name=str(campaign_name) if campaign_name is not None else None,
        message=message,
    )
