"""Planner configuration for echo-agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import asyncio

from penguiflow.catalog import build_catalog
from penguiflow.artifacts import ArtifactRetentionConfig, InMemoryArtifactStore
from penguiflow.planner import ReactPlanner
from penguiflow.planner.memory import MemoryBudget, MemoryIsolation, ShortTermMemoryConfig
from penguiflow.rich_output import DEFAULT_ALLOWLIST, RichOutputConfig, attach_rich_output_nodes, get_runtime

from .config import Config
from .tools import build_catalog_bundle


SYSTEM_PROMPT_EXTRA = """You are assistant for the ads operation team.

Available Tools:
- Check Campaign Line Status: Look up the status of a specific campaign line item by ID from Databricks
- Search Segment: Perform semantic search over audience segments using keyword queries

Capabilities:
- You can query the status of campaign line items
- You can search for audience segments based on descriptions and keywords

Limitations:
- You cannot make real-time changes to live advertising campaigns
- You have read-only access to campaign and segment data
- You cannot access data outside of the provided datasets

Rules:
- If the user asks for audience/segment search or mentions wanting a segment, call `search_segment`
- If the user asks for status of a particular line item and provides a line ID, call `check_campaign_line_status`
- Always provide context for tool usage and explain what data you're retrieving
- Be transparent about read-only access and data limitations"""


@dataclass
class PlannerBundle:
    """Container for planner and registry."""

    planner: ReactPlanner
    registry: Any


def _build_system_prompt(config: Config, rich_output_prompt: str = "") -> str:
    """Build the system prompt, optionally including memory and rich output context."""
    prompt = SYSTEM_PROMPT_EXTRA
    if rich_output_prompt:
        prompt = prompt + "\n\n" + rich_output_prompt
    return prompt


def _build_rich_output_config(config: Config) -> RichOutputConfig:
    """Build rich output config for component rendering."""
    allowlist = config.rich_output_allowlist or list(DEFAULT_ALLOWLIST)
    return RichOutputConfig(
        enabled=config.rich_output_enabled,
        allowlist=allowlist,
        include_prompt_catalog=config.rich_output_include_prompt_catalog,
        include_prompt_examples=config.rich_output_include_prompt_examples,
        max_payload_bytes=config.rich_output_max_payload_bytes,
        max_total_bytes=config.rich_output_max_total_bytes,
    )


def _build_short_term_memory(config: Config) -> ShortTermMemoryConfig | None:
    """Build built-in short-term memory configuration."""
    if not config.short_term_memory_enabled:
        return None

    budget = MemoryBudget(
        full_zone_turns=config.short_term_memory_full_zone_turns,
        summary_max_tokens=config.short_term_memory_summary_max_tokens,
        total_max_tokens=config.short_term_memory_total_max_tokens,
        overflow_policy=config.short_term_memory_overflow_policy,
    )
    isolation = MemoryIsolation(
        tenant_key=config.short_term_memory_tenant_key,
        user_key=config.short_term_memory_user_key,
        session_key=config.short_term_memory_session_key,
        require_explicit_key=config.short_term_memory_require_explicit_key,
    )

    return ShortTermMemoryConfig(
        strategy=config.short_term_memory_strategy,
        budget=budget,
        isolation=isolation,
        summarizer_model=config.short_term_memory_summarizer_model,
        include_trajectory_digest=config.short_term_memory_include_trajectory_digest,
        recovery_backlog_limit=config.short_term_memory_recovery_backlog_limit,
        retry_attempts=config.short_term_memory_retry_attempts,
        retry_backoff_base_s=config.short_term_memory_retry_backoff_base_s,
        degraded_retry_interval_s=config.short_term_memory_degraded_retry_interval_s,
    )



def _build_artifact_store(config: Config) -> InMemoryArtifactStore | None:
    """Build artifact store from config.

    When disabled, ReactPlanner falls back to NoOpArtifactStore.
    """
    if not config.artifact_store_enabled:
        return None

    return InMemoryArtifactStore(
        retention=ArtifactRetentionConfig(
            ttl_seconds=config.artifact_store_ttl_seconds,
            max_artifact_bytes=config.artifact_store_max_artifact_bytes,
            max_session_bytes=config.artifact_store_max_session_bytes,
            max_trace_bytes=config.artifact_store_max_trace_bytes,
            max_artifacts_per_trace=config.artifact_store_max_artifacts_per_trace,
            max_artifacts_per_session=config.artifact_store_max_artifacts_per_session,
            cleanup_strategy=config.artifact_store_cleanup_strategy,
        ),
    )


def build_planner(config: Config, *, event_callback=None) -> PlannerBundle:
    """Build ReactPlanner with generated tool catalog."""

    nodes, registry = build_catalog_bundle()
    rich_output_config = _build_rich_output_config(config)
    nodes.extend(attach_rich_output_nodes(registry, config=rich_output_config))
    catalog = build_catalog(nodes, registry)


    reflection_cfg = None

    summarizer_llm = None

    reflection_llm = None

    artifact_store = _build_artifact_store(config)

    rich_output_prompt = get_runtime().prompt_section()

    planner = ReactPlanner(
        llm=config.llm_model,
        catalog=catalog,
        system_prompt_extra=_build_system_prompt(config, rich_output_prompt),
        max_iters=config.planner_max_iters,
        hop_budget=config.planner_hop_budget,
        absolute_max_parallel=config.planner_absolute_max_parallel,
        multi_action_sequential=config.planner_multi_action_sequential,
        multi_action_read_only_only=config.planner_multi_action_read_only_only,
        multi_action_max_tools=config.planner_multi_action_max_tools,
        summarizer_llm=summarizer_llm,
        reflection_config=reflection_cfg,
        reflection_llm=reflection_llm,
        planning_hints=None,
        artifact_store=artifact_store,
        event_callback=event_callback,
        stream_final_response=config.planner_stream_final_response,
        short_term_memory=_build_short_term_memory(config),
        use_native_reasoning=True,
        reasoning_effort="medium",   # common values: "low" | "medium" | "high"
    )

    return PlannerBundle(planner=planner, registry=registry)
