import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

PLATFORM = "platform"
NODE = "node"
metadata = sa.MetaData()

COMMAND_KINDS = ("submit", "cancel", "modify", "cancel_all", "close_position", "rebalance")
COMMAND_STATES = ("pending", "claimed", "sent", "done", "denied", "failed", "unknown")

order_commands = sa.Table(
    "order_commands",
    metadata,
    sa.Column("id", UUID(as_uuid=True), primary_key=True),
    sa.Column("created", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    sa.Column("kind", sa.String(24), nullable=False),
    sa.Column("payload", JSONB, nullable=False),
    sa.Column("source", sa.String(24), nullable=False, server_default="manual"),
    sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
    sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
    sa.Column("claimed_by", sa.String(64)),
    sa.Column("claimed_at", sa.DateTime(timezone=True)),
    sa.Column("client_order_id", sa.String(64), unique=True),
    sa.Column("result", JSONB),
    sa.Column("error", sa.Text),
    sa.CheckConstraint(f"kind in {COMMAND_KINDS}", name="command_kind"),
    sa.CheckConstraint(f"status in {COMMAND_STATES}", name="command_status"),
    sa.Index("order_commands_pending", "created", postgresql_where=sa.text("status = 'pending'")),
    schema=PLATFORM,
)

order_command_events = sa.Table(
    "order_command_events",
    metadata,
    sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
    sa.Column(
        "command_id",
        UUID(as_uuid=True),
        sa.ForeignKey(f"{PLATFORM}.order_commands.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    sa.Column("time", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    sa.Column("status", sa.String(16), nullable=False),
    sa.Column("detail", JSONB),
    schema=PLATFORM,
)

node_orders = sa.Table(
    "orders",
    metadata,
    sa.Column("client_order_id", sa.String(64), primary_key=True),
    sa.Column("venue_order_id", sa.String(64), index=True),
    sa.Column("command_id", UUID(as_uuid=True), index=True),
    sa.Column("primary_id", sa.String(64), index=True),
    sa.Column("exec_algorithm", sa.String(32)),
    sa.Column("strategy_id", sa.String(64)),
    sa.Column("instrument_id", sa.String(64), nullable=False),
    sa.Column("side", sa.String(8), nullable=False),
    sa.Column("order_type", sa.String(32), nullable=False),
    sa.Column("time_in_force", sa.String(16)),
    sa.Column("quantity", sa.Numeric, nullable=False),
    sa.Column("filled_qty", sa.Numeric, nullable=False, server_default="0"),
    sa.Column("avg_px", sa.Numeric),
    sa.Column("price", sa.Numeric),
    sa.Column("trigger_price", sa.Numeric),
    sa.Column("status", sa.String(24), nullable=False),
    sa.Column("is_open", sa.Boolean, nullable=False),
    sa.Column("is_closed", sa.Boolean, nullable=False),
    sa.Column("reconciled", sa.Boolean, nullable=False, server_default="false"),
    sa.Column("created", sa.DateTime(timezone=True), nullable=False),
    sa.Column("updated", sa.DateTime(timezone=True), nullable=False),
    sa.Column("last_event", sa.String(40)),
    sa.Column("reason", sa.Text),
    schema=NODE,
)

node_fills = sa.Table(
    "fills",
    metadata,
    sa.Column("trade_id", sa.String(64), primary_key=True),
    sa.Column("client_order_id", sa.String(64), nullable=False, index=True),
    sa.Column("venue_order_id", sa.String(64)),
    sa.Column("instrument_id", sa.String(64), nullable=False),
    sa.Column("side", sa.String(8), nullable=False),
    sa.Column("quantity", sa.Numeric, nullable=False),
    sa.Column("price", sa.Numeric, nullable=False),
    sa.Column("commission", sa.Numeric),
    sa.Column("commission_currency", sa.String(16)),
    sa.Column("liquidity_side", sa.String(8)),
    sa.Column("time", sa.DateTime(timezone=True), nullable=False),
    schema=NODE,
)

node_positions = sa.Table(
    "positions",
    metadata,
    sa.Column("instrument_id", sa.String(64), primary_key=True),
    sa.Column("side", sa.String(8), nullable=False),
    sa.Column("quantity", sa.Numeric, nullable=False),
    sa.Column("avg_px_open", sa.Numeric),
    sa.Column("realized_pnl", sa.Numeric),
    sa.Column("updated", sa.DateTime(timezone=True), nullable=False),
    schema=NODE,
)

node_status = sa.Table(
    "status",
    metadata,
    sa.Column("name", sa.String(32), primary_key=True),
    sa.Column("value", JSONB, nullable=False),
    sa.Column("updated", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    schema=NODE,
)

node_reconciliation = sa.Table(
    "reconciliation",
    metadata,
    sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
    sa.Column("time", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    sa.Column("kind", sa.String(32), nullable=False),
    sa.Column("severity", sa.String(16), nullable=False),
    sa.Column("instrument_id", sa.String(64)),
    sa.Column("client_order_id", sa.String(64)),
    sa.Column("detail", sa.Text, nullable=False),
    schema=NODE,
)

trading_state = sa.Table(
    "trading_state",
    metadata,
    sa.Column("id", sa.SmallInteger, primary_key=True, server_default="1"),
    sa.Column("halted", sa.Boolean, nullable=False, server_default="false"),
    sa.Column("reason", sa.Text),
    sa.Column("updated", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    sa.CheckConstraint("id = 1", name="one_trading_state"),
    schema=PLATFORM,
)

compliance = sa.Table(
    "compliance",
    metadata,
    sa.Column("symbol", sa.String(32), primary_key=True),
    sa.Column("allowed", sa.Boolean, nullable=False),
    sa.Column("reason", sa.Text, nullable=False),
    sa.Column("as_of", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    schema=PLATFORM,
)

STAGES = ("research", "validated", "shadow", "canary", "live", "retired")
DEFINITION_SOURCES = ("template", "user", "import")

strategy_definitions = sa.Table(
    "strategy_definitions",
    metadata,
    sa.Column("name", sa.String(64), primary_key=True),
    sa.Column("current_version", sa.Integer, nullable=False),
    sa.Column("archived", sa.Boolean, nullable=False, server_default=sa.false()),
    sa.Column("source", sa.String(16), nullable=False),
    sa.Column("created", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    sa.Column("updated", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    sa.CheckConstraint(f"source in {DEFINITION_SOURCES}", name="definition_source"),
    schema=PLATFORM,
)

TRIAL_ORIGINS = ("version", "preview")

strategy_trials = sa.Table(
    "strategy_trials",
    metadata,
    sa.Column("rule_hash", sa.String(64), primary_key=True),
    sa.Column("origin", sa.String(16), nullable=False),
    sa.Column("strategy", sa.String(64)),
    sa.Column(
        "first_seen", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    ),
    sa.CheckConstraint(f"origin in {TRIAL_ORIGINS}", name="trial_origin"),
    schema=PLATFORM,
)

strategy_definition_versions = sa.Table(
    "strategy_definition_versions",
    metadata,
    sa.Column(
        "name",
        sa.String(64),
        sa.ForeignKey(f"{PLATFORM}.strategy_definitions.name"),
        primary_key=True,
    ),
    sa.Column("version", sa.Integer, primary_key=True),
    sa.Column("spec", JSONB, nullable=False),
    sa.Column("spec_hash", sa.String(64), nullable=False),
    sa.Column("note", sa.Text),
    sa.Column("created", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    schema=PLATFORM,
)

strategy_versions = sa.Table(
    "strategy_versions",
    metadata,
    sa.Column("id", UUID(as_uuid=True), primary_key=True),
    sa.Column("strategy", sa.String(64), nullable=False, index=True),
    sa.Column("code_hash", sa.String(64), nullable=False),
    sa.Column("git_sha", sa.String(40)),
    sa.Column("params", JSONB, nullable=False),
    sa.Column("engine", sa.String(32), nullable=False),
    sa.Column("data_start", sa.Date),
    sa.Column("data_end", sa.Date),
    sa.Column("metrics", JSONB, nullable=False, server_default="{}"),
    sa.Column("created", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    sa.UniqueConstraint("strategy", "code_hash", name="one_version_per_code"),
    schema=PLATFORM,
)

strategies = sa.Table(
    "strategies",
    metadata,
    sa.Column("name", sa.String(64), primary_key=True),
    sa.Column("stage", sa.String(16), nullable=False, server_default="research"),
    sa.Column(
        "version_id",
        UUID(as_uuid=True),
        sa.ForeignKey(f"{PLATFORM}.strategy_versions.id"),
        nullable=False,
    ),
    sa.Column("capital", sa.Float, nullable=False, server_default="0"),
    sa.Column("stage_since", sa.DateTime(timezone=True), nullable=False),
    sa.Column("updated", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    sa.CheckConstraint(f"stage in {STAGES}", name="strategy_stage"),
    schema=PLATFORM,
)

strategy_decisions = sa.Table(
    "strategy_decisions",
    metadata,
    sa.Column("strategy", sa.String(64), primary_key=True),
    sa.Column("session", sa.Date, primary_key=True),
    sa.Column(
        "version_id",
        UUID(as_uuid=True),
        sa.ForeignKey(f"{PLATFORM}.strategy_versions.id"),
        nullable=False,
    ),
    sa.Column("weights", JSONB, nullable=False),
    sa.Column("recorded", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    schema=PLATFORM,
)

lifecycle_events = sa.Table(
    "lifecycle_events",
    metadata,
    sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
    sa.Column("strategy", sa.String(64), nullable=False, index=True),
    sa.Column("time", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    sa.Column("from_stage", sa.String(16)),
    sa.Column("to_stage", sa.String(16), nullable=False),
    sa.Column("version_id", UUID(as_uuid=True), nullable=False),
    sa.Column("actor", sa.String(16), nullable=False),
    sa.Column("reason", sa.Text, nullable=False),
    sa.Column("gates", JSONB, nullable=False, server_default="[]"),
    schema=PLATFORM,
)

TARGET_SOURCES = ("node", "api")

strategy_targets = sa.Table(
    "strategy_targets",
    metadata,
    sa.Column("strategy", sa.String(64), primary_key=True),
    sa.Column("source", sa.String(8), nullable=False),
    sa.Column("stage", sa.String(16), nullable=False),
    sa.Column("allocation", sa.Float, nullable=False),
    sa.Column("weights", JSONB, nullable=False),
    sa.Column("version_id", sa.String(64)),
    sa.Column("session", sa.Date),
    sa.Column("updated", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    sa.CheckConstraint(f"source in {TARGET_SOURCES}", name="target_source"),
    schema=PLATFORM,
)

arbiter_runs = sa.Table(
    "arbiter_runs",
    metadata,
    sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
    sa.Column("time", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    sa.Column("equity", sa.Float, nullable=False),
    sa.Column("targets", JSONB, nullable=False),
    sa.Column("weights", JSONB, nullable=False),
    sa.Column("notes", JSONB, nullable=False),
    sa.Column("orders", JSONB, nullable=False),
    sa.Column("command_id", UUID(as_uuid=True)),
    sa.Column("detail", sa.Text),
    schema=PLATFORM,
)

shadow_orders = sa.Table(
    "shadow_orders",
    metadata,
    sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
    sa.Column("time", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    sa.Column("strategy", sa.String(64), nullable=False, index=True),
    sa.Column("version_id", sa.String(64)),
    sa.Column("symbol", sa.String(32), nullable=False),
    sa.Column("side", sa.String(4), nullable=False),
    sa.Column("qty", sa.Numeric, nullable=False),
    sa.Column("price", sa.Numeric, nullable=False),
    sa.Column("source", sa.String(16), nullable=False),
    schema=PLATFORM,
)

autopilot = sa.Table(
    "autopilot",
    metadata,
    sa.Column("id", sa.SmallInteger, primary_key=True, server_default="1"),
    sa.Column("mode", sa.String(16), nullable=False),
    sa.Column("allocation", sa.Float, nullable=False),
    sa.Column("crypto", sa.Boolean, nullable=False),
    sa.Column("managed", JSONB, nullable=False, server_default="[]"),
    sa.Column("updated", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    sa.CheckConstraint("id = 1", name="one_autopilot"),
    schema=PLATFORM,
)

BREAKER_LEVELS = ("trade", "strategy", "portfolio", "system")
BREAKER_STATES = ("closed", "open", "half_open")

breakers = sa.Table(
    "breakers",
    metadata,
    sa.Column("name", sa.String(96), primary_key=True),
    sa.Column("level", sa.String(16), nullable=False),
    sa.Column("state", sa.String(16), nullable=False, server_default="closed"),
    sa.Column("reason", sa.Text),
    sa.Column("opened_at", sa.DateTime(timezone=True)),
    sa.Column("retry_at", sa.DateTime(timezone=True)),
    sa.Column("cooldown_seconds", sa.Integer, nullable=False, server_default="86400"),
    sa.Column("trips", sa.Integer, nullable=False, server_default="0"),
    sa.Column("updated", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    sa.CheckConstraint(f"level in {BREAKER_LEVELS}", name="breaker_level"),
    sa.CheckConstraint(f"state in {BREAKER_STATES}", name="breaker_state"),
    schema=PLATFORM,
)

breaker_events = sa.Table(
    "breaker_events",
    metadata,
    sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
    sa.Column("time", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    sa.Column("name", sa.String(96), nullable=False, index=True),
    sa.Column("level", sa.String(16), nullable=False),
    sa.Column("from_state", sa.String(16), nullable=False),
    sa.Column("to_state", sa.String(16), nullable=False),
    sa.Column("actor", sa.String(16), nullable=False),
    sa.Column("reason", sa.Text, nullable=False),
    schema=PLATFORM,
)

KILL_SCOPES = ("global", "account", "strategy")

kill_switches = sa.Table(
    "kill_switches",
    metadata,
    sa.Column("name", sa.String(96), primary_key=True),
    sa.Column("scope", sa.String(16), nullable=False),
    sa.Column("engaged", sa.Boolean, nullable=False),
    sa.Column("reason", sa.Text),
    sa.Column("actor", sa.String(32), nullable=False),
    sa.Column("updated", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    sa.CheckConstraint(f"scope in {KILL_SCOPES}", name="kill_scope"),
    schema=PLATFORM,
)

MODEL_TYPES = ("rl_agent", "classifier", "llm")
DEPLOYMENT_STAGES = ("staging", "production", "archived", "rolled_back")

models = sa.Table(
    "models",
    metadata,
    sa.Column("model_id", sa.String(96), primary_key=True),
    sa.Column("name", sa.String(64), nullable=False, index=True),
    sa.Column("model_type", sa.String(16), nullable=False),
    sa.Column("strategy", sa.String(64)),
    sa.Column("training_data_version", sa.Text, nullable=False),
    sa.Column("hyperparameters", JSONB, nullable=False),
    sa.Column("artifact_path", sa.Text, nullable=False),
    sa.Column("artifact_hash", sa.String(64)),
    sa.Column("feature_version", JSONB, nullable=False, server_default="{}"),
    sa.Column("metrics", JSONB, nullable=False, server_default="{}"),
    sa.Column("deployment_stage", sa.String(16), nullable=False, server_default="staging"),
    sa.Column("promoted_at", sa.DateTime(timezone=True)),
    sa.Column("promoted_by", sa.String(64)),
    sa.Column("promotion_reason", sa.Text),
    sa.Column("created", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    sa.CheckConstraint(f"model_type in {MODEL_TYPES}", name="model_type"),
    sa.CheckConstraint(f"deployment_stage in {DEPLOYMENT_STAGES}", name="model_stage"),
    schema=PLATFORM,
)

features = sa.Table(
    "features",
    metadata,
    sa.Column("name", sa.String(64), primary_key=True),
    sa.Column("version", sa.Integer, primary_key=True, server_default="1"),
    sa.Column("entity", sa.String(32), nullable=False, server_default="symbol"),
    sa.Column("sources", JSONB, nullable=False, server_default="[]"),
    sa.Column("owner", sa.String(32), nullable=False, server_default=""),
    sa.Column("lineage", sa.String(16), nullable=False, server_default=""),
    sa.Column("description", sa.Text, nullable=False),
    sa.Column("computation", sa.Text, nullable=False),
    sa.Column("ttl_seconds", sa.Integer, nullable=False),
    sa.Column("expected_min", sa.Float),
    sa.Column("expected_max", sa.Float),
    sa.Column("offline_table", sa.String(64), nullable=False),
    sa.Column("updated", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    schema=PLATFORM,
)

FEATURE_TABLES = {}
for _name in ("alpha_factors", "sentiment_cache", "obi_metrics"):
    FEATURE_TABLES[_name] = sa.Table(
        _name,
        metadata,
        sa.Column("feature", sa.String(64), primary_key=True),
        sa.Column("version", sa.Integer, primary_key=True, server_default="1"),
        sa.Column("symbol", sa.String(32), primary_key=True),
        sa.Column("time", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("value", sa.Float, nullable=False),
        sa.Column("quality_flags", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "knowledge_ts", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        schema=PLATFORM,
    )

FAILURE_TYPES = ("none", "technical", "statistical")

strategy_health = sa.Table(
    "strategy_health",
    metadata,
    sa.Column("subject", sa.String(96), primary_key=True),
    sa.Column("failure_type", sa.String(16), nullable=False),
    sa.Column("factor", sa.Float, nullable=False),
    sa.Column("reason", sa.Text, nullable=False),
    sa.Column("since", sa.DateTime(timezone=True), nullable=False),
    sa.Column("updated", sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint(f"failure_type in {FAILURE_TYPES}", name="failure_type"),
    sa.CheckConstraint("factor >= 0 and factor <= 1", name="health_factor"),
    schema=PLATFORM,
)

DRIFT_KINDS = ("technical", "statistical")
DRIFT_SEVERITIES = ("low", "medium", "high")

model_drift = sa.Table(
    "model_drift",
    metadata,
    sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
    sa.Column("time", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    sa.Column("subject", sa.String(96), nullable=False, index=True),
    sa.Column("kind", sa.String(16), nullable=False),
    sa.Column("metric", sa.String(32), nullable=False),
    sa.Column("value", sa.Float),
    sa.Column("threshold", sa.Float),
    sa.Column("drift_severity", sa.String(8), nullable=False),
    sa.Column("detail", sa.Text, nullable=False),
    sa.CheckConstraint(f"kind in {DRIFT_KINDS}", name="drift_kind"),
    sa.CheckConstraint(f"drift_severity in {DRIFT_SEVERITIES}", name="drift_severity"),
    schema=PLATFORM,
)

data_quality = sa.Table(
    "data_quality",
    metadata,
    sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
    sa.Column("time", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    sa.Column("source", sa.String(16), nullable=False),
    sa.Column("symbol", sa.String(32), nullable=False, index=True),
    sa.Column("session", sa.Date, nullable=False),
    sa.Column("quality_flags", sa.Integer, nullable=False),
    sa.Column("reason_code", sa.String(32), nullable=False),
    sa.Column("detail", sa.Text, nullable=False),
    sa.Column("previous_digest", sa.String(64), nullable=False),
    sa.Column("digest", sa.String(64), nullable=False, unique=True),
    sa.UniqueConstraint("source", "symbol", "session", "reason_code", name="one_finding"),
    schema=PLATFORM,
)

daily_bars = sa.Table(
    "daily_bars",
    metadata,
    sa.Column("symbol", sa.String(32), primary_key=True),
    sa.Column("session", sa.Date, primary_key=True),
    sa.Column("open", sa.Float),
    sa.Column("close", sa.Float),
    sa.Column("source", sa.String(16), nullable=False),
    sa.Column("quality_flags", sa.Integer, nullable=False, server_default="0"),
    sa.Column("updated", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    schema=PLATFORM,
)
