from prometheus_client import Counter, Gauge, start_http_server

ORDERS = Counter("algotr_node_order_events", "Order events the node saw", ["status"])
ARBITER_RUNS = Counter("algotr_arbiter_runs", "Arbitrations the node ran", ["sent"])
HEARTBEAT = Gauge("algotr_node_heartbeat_seconds", "Unix time of the node's last heartbeat")


def serve(port: int) -> None:
    """Expose the node's metrics for Prometheus to scrape."""
    start_http_server(port)
