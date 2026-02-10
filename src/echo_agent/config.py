"""Configuration for echo-agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_flag(name: str, default: bool) -> bool:
    """Parse boolean from environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Parse integer from environment variable."""
    raw = os.getenv(name)
    return int(raw) if raw is not None else default


def _env_float(name: str, default: float) -> float:
    """Parse float from environment variable."""
    raw = os.getenv(name)
    return float(raw) if raw is not None else default


def _env_str(name: str, default: str) -> str:
    """Parse string from environment variable."""
    raw = os.getenv(name)
    return raw if raw is not None else default


def _env_csv(name: str, default: list[str]) -> list[str]:
    """Parse comma-separated list from environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items


def _env_optional_str(name: str, default: str | None) -> str | None:
    """Parse optional string from environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw or None


@dataclass
class Config:
    """Environment-driven configuration.

    All settings can be overridden via environment variables.
    Use Config.from_env() to load from environment.
    """

    # LLM Configuration
    llm_model: str = "databricks/databricks-claude-3-7-sonnet"
    summarizer_model: str | None = None
    reflection_model: str | None = None

    # Feature Flags
    memory_enabled: bool = False
    summarizer_enabled: bool = False
    reflection_enabled: bool = False
    short_term_memory_enabled: bool = False

    # Reflection Settings
    reflection_quality_threshold: float = 0.8
    reflection_max_revisions: int = 2

    # Service URLs
    memory_base_url: str | None = None
    rag_server_base_url: str | None = None
    wayfinder_base_url: str | None = None

    # Planner Settings
    planner_max_iters: int = 3
    planner_hop_budget: int = 3
    planner_absolute_max_parallel: int = 1
    planner_stream_final_response: bool = False
    planner_multi_action_sequential: bool = False
    planner_multi_action_read_only_only: bool = True
    planner_multi_action_max_tools: int = 2

    # Artifact Store (binary/large text)
    artifact_store_enabled: bool = False
    artifact_store_ttl_seconds: int = 3600
    artifact_store_max_artifact_bytes: int = 52428800
    artifact_store_max_session_bytes: int = 524288000
    artifact_store_max_trace_bytes: int = 104857600
    artifact_store_max_artifacts_per_trace: int = 100
    artifact_store_max_artifacts_per_session: int = 1000
    artifact_store_cleanup_strategy: str = "lru"

    # Rich Output (Component Artifacts)
    rich_output_enabled: bool = False
    rich_output_allowlist: list[str] = field(default_factory=lambda: ['markdown', 'json', 'echarts', 'mermaid', 'plotly', 'datagrid', 'metric', 'report', 'grid', 'tabs', 'accordion', 'code', 'latex', 'callout', 'image', 'video', 'form', 'confirm', 'select_option'])
    rich_output_include_prompt_catalog: bool = True
    rich_output_include_prompt_examples: bool = False
    rich_output_max_payload_bytes: int = 250000
    rich_output_max_total_bytes: int = 2000000

    # Built-in Short-Term Memory (ReactPlanner)
    short_term_memory_strategy: str = 'none'
    short_term_memory_full_zone_turns: int = 5
    short_term_memory_summary_max_tokens: int = 1000
    short_term_memory_total_max_tokens: int = 10000
    short_term_memory_overflow_policy: str = 'truncate_oldest'
    short_term_memory_tenant_key: str = 'tenant_id'
    short_term_memory_user_key: str = 'user_id'
    short_term_memory_session_key: str = 'session_id'
    short_term_memory_require_explicit_key: bool = True
    short_term_memory_include_trajectory_digest: bool = True
    short_term_memory_summarizer_model: str | None = None
    short_term_memory_recovery_backlog_limit: int = 20
    short_term_memory_retry_attempts: int = 3
    short_term_memory_retry_backoff_base_s: float = 2.0
    short_term_memory_degraded_retry_interval_s: float = 30.0


    @classmethod
    def from_env(cls) -> Config:
        """Load configuration from environment variables."""
        return cls(
            # LLM Configuration
            llm_model=os.getenv("LLM_MODEL", "databricks/databricks-claude-3-7-sonnet"),
            summarizer_model=os.getenv("SUMMARIZER_MODEL"),
            reflection_model=os.getenv("REFLECTION_MODEL"),

            # Feature Flags
            memory_enabled=_env_flag("MEMORY_ENABLED", False),
            summarizer_enabled=_env_flag("SUMMARIZER_ENABLED", False),
            reflection_enabled=_env_flag("REFLECTION_ENABLED", False),
            short_term_memory_enabled=_env_flag("SHORT_TERM_MEMORY_ENABLED", False),

            # Reflection Settings
            reflection_quality_threshold=_env_float("REFLECTION_QUALITY_THRESHOLD", 0.8),
            reflection_max_revisions=_env_int("REFLECTION_MAX_REVISIONS", 2),

            # Service URLs (None if not set)
            memory_base_url=os.getenv("MEMORY_BASE_URL"),
            rag_server_base_url=os.getenv("RAG_SERVER_BASE_URL"),
            wayfinder_base_url=os.getenv("WAYFINDER_BASE_URL"),

            # Planner Settings
            planner_max_iters=_env_int("PLANNER_MAX_ITERS", 3),
            planner_hop_budget=_env_int("PLANNER_HOP_BUDGET", 3),
            planner_absolute_max_parallel=_env_int("PLANNER_ABSOLUTE_MAX_PARALLEL", 1),
            planner_stream_final_response=_env_flag("PLANNER_STREAM_FINAL_RESPONSE", False),
            planner_multi_action_sequential=_env_flag(
                "PLANNER_MULTI_ACTION_SEQUENTIAL",
                False,
            ),
            planner_multi_action_read_only_only=_env_flag(
                "PLANNER_MULTI_ACTION_READ_ONLY_ONLY",
                True,
            ),
            planner_multi_action_max_tools=_env_int(
                "PLANNER_MULTI_ACTION_MAX_TOOLS",
                2,
            ),

            # Artifact Store (binary/large text)
            artifact_store_enabled=_env_flag("ARTIFACT_STORE_ENABLED", False),
            artifact_store_ttl_seconds=_env_int("ARTIFACT_STORE_TTL_SECONDS", 3600),
            artifact_store_max_artifact_bytes=_env_int(
                "ARTIFACT_STORE_MAX_ARTIFACT_BYTES",
                52428800,
            ),
            artifact_store_max_session_bytes=_env_int(
                "ARTIFACT_STORE_MAX_SESSION_BYTES",
                524288000,
            ),
            artifact_store_max_trace_bytes=_env_int(
                "ARTIFACT_STORE_MAX_TRACE_BYTES",
                104857600,
            ),
            artifact_store_max_artifacts_per_trace=_env_int(
                "ARTIFACT_STORE_MAX_ARTIFACTS_PER_TRACE",
                100,
            ),
            artifact_store_max_artifacts_per_session=_env_int(
                "ARTIFACT_STORE_MAX_ARTIFACTS_PER_SESSION",
                1000,
            ),
            artifact_store_cleanup_strategy=_env_str(
                "ARTIFACT_STORE_CLEANUP_STRATEGY",
                "lru",
            ),

            # Rich Output (Component Artifacts)
            rich_output_enabled=_env_flag("RICH_OUTPUT_ENABLED", False),
            rich_output_allowlist=_env_csv(
                "RICH_OUTPUT_ALLOWLIST",
                ['markdown', 'json', 'echarts', 'mermaid', 'plotly', 'datagrid', 'metric', 'report', 'grid', 'tabs', 'accordion', 'code', 'latex', 'callout', 'image', 'video', 'form', 'confirm', 'select_option'],
            ),
            rich_output_include_prompt_catalog=_env_flag(
                "RICH_OUTPUT_INCLUDE_PROMPT_CATALOG",
                True,
            ),
            rich_output_include_prompt_examples=_env_flag(
                "RICH_OUTPUT_INCLUDE_PROMPT_EXAMPLES",
                False,
            ),
            rich_output_max_payload_bytes=_env_int(
                "RICH_OUTPUT_MAX_PAYLOAD_BYTES",
                250000,
            ),
            rich_output_max_total_bytes=_env_int(
                "RICH_OUTPUT_MAX_TOTAL_BYTES",
                2000000,
            ),

            # Built-in Short-Term Memory
            short_term_memory_strategy=_env_str("SHORT_TERM_MEMORY_STRATEGY", 'none'),
            short_term_memory_full_zone_turns=_env_int(
                "SHORT_TERM_MEMORY_FULL_ZONE_TURNS",
                5,
            ),
            short_term_memory_summary_max_tokens=_env_int(
                "SHORT_TERM_MEMORY_SUMMARY_MAX_TOKENS",
                1000,
            ),
            short_term_memory_total_max_tokens=_env_int(
                "SHORT_TERM_MEMORY_TOTAL_MAX_TOKENS",
                10000,
            ),
            short_term_memory_overflow_policy=_env_str(
                "SHORT_TERM_MEMORY_OVERFLOW_POLICY",
                'truncate_oldest',
            ),
            short_term_memory_tenant_key=_env_str(
                "SHORT_TERM_MEMORY_TENANT_KEY",
                'tenant_id',
            ),
            short_term_memory_user_key=_env_str(
                "SHORT_TERM_MEMORY_USER_KEY",
                'user_id',
            ),
            short_term_memory_session_key=_env_str(
                "SHORT_TERM_MEMORY_SESSION_KEY",
                'session_id',
            ),
            short_term_memory_require_explicit_key=_env_flag(
                "SHORT_TERM_MEMORY_REQUIRE_EXPLICIT_KEY",
                True,
            ),
            short_term_memory_include_trajectory_digest=_env_flag(
                "SHORT_TERM_MEMORY_INCLUDE_TRAJECTORY_DIGEST",
                True,
            ),
            short_term_memory_summarizer_model=_env_optional_str(
                "SHORT_TERM_MEMORY_SUMMARIZER_MODEL",
                None,
            ),
            short_term_memory_recovery_backlog_limit=_env_int(
                "SHORT_TERM_MEMORY_RECOVERY_BACKLOG_LIMIT",
                20,
            ),
            short_term_memory_retry_attempts=_env_int(
                "SHORT_TERM_MEMORY_RETRY_ATTEMPTS",
                3,
            ),
            short_term_memory_retry_backoff_base_s=_env_float(
                "SHORT_TERM_MEMORY_RETRY_BACKOFF_BASE_S",
                2.0,
            ),
            short_term_memory_degraded_retry_interval_s=_env_float(
                "SHORT_TERM_MEMORY_DEGRADED_RETRY_INTERVAL_S",
                30.0,
            ),
        )
