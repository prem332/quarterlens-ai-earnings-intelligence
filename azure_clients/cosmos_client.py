"""
Decision Log document schema:
{
    "id":             str  — uuid4
    "run_id":         str  — analysis run identifier (groups all agent logs for one request)
    "agent":          str  — agent name (e.g. "numeric_validation_agent")
    "timestamp":      str  — ISO 8601 UTC
    "tool_called":    str  — tool function name
    "tool_args":      dict — arguments passed to the tool
    "result_summary": str  — human-readable outcome
    "confidence":     float — agent confidence score (0.0–1.0)
    "token_cost":     int  — tokens consumed by this agent step
    "latency_ms":     int  — wall-clock latency for this step
    "status":         str  — "success" | "error"
}
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from azure.cosmos import CosmosClient, PartitionKey, exceptions
from azure_clients.key_vault_client import kv

logger = logging.getLogger(__name__)

DATABASE_NAME = "quarterlens"
CONTAINER_NAME = "decision_log"
PARTITION_KEY = "/run_id"


class CosmosDecisionLogClient:
    """
    Writes and queries Decision Log entries in Cosmos DB NoSQL API.
    Partition key is run_id — all agent steps for one analysis run
    are co-located, making per-run queries cheap.
    """

    def __init__(self):
        uri = kv.get_secret("AZURE-COSMOS-URI")
        key = kv.get_secret("AZURE-COSMOS-KEY")

        self._client = CosmosClient(url=uri, credential=key)
        self._container = self._get_or_create_container()
        logger.info(
            "CosmosDecisionLogClient: connected to %s/%s",
            DATABASE_NAME, CONTAINER_NAME,
        )

    def _get_or_create_container(self):
        """Idempotent — safe to call on every startup."""
        db = self._client.create_database_if_not_exists(DATABASE_NAME)
        container = db.create_container_if_not_exists(
            id=CONTAINER_NAME,
            partition_key=PartitionKey(path=PARTITION_KEY),
        )
        return container

    def log(
        self,
        run_id: str,
        agent: str,
        tool_called: str,
        result_summary: str,
        status: str,
        tool_args: Optional[dict] = None,
        confidence: Optional[float] = None,
        token_cost: Optional[int] = None,
        latency_ms: Optional[int] = None,
    ) -> str:
        """
        Write one Decision Log entry.

        Args:
            run_id:         Analysis run identifier (groups all steps for one request).
            agent:          Agent name, e.g. "numeric_validation_agent".
            tool_called:    Tool function name, e.g. "calculate_metric".
            result_summary: Human-readable outcome.
            status:         "success" or "error".
            tool_args:      Arguments passed to the tool (optional).
            confidence:     Agent confidence score 0.0–1.0 (optional).
            token_cost:     Tokens consumed (optional).
            latency_ms:     Wall-clock latency for this step (optional).

        Returns:
            Document id of the created entry.
        """
        doc_id = str(uuid.uuid4())
        document = {
            "id": doc_id,
            "run_id": run_id,
            "agent": agent,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_called": tool_called,
            "tool_args": tool_args or {},
            "result_summary": result_summary,
            "confidence": confidence,
            "token_cost": token_cost,
            "latency_ms": latency_ms,
            "status": status,
        }

        self._container.create_item(body=document)
        logger.debug(
            "CosmosDecisionLogClient: logged [%s] %s → %s (run=%s)",
            status, agent, tool_called, run_id,
        )
        return doc_id

    def get_run_log(self, run_id: str) -> list[dict]:
        """
        Fetch all Decision Log entries for a given run, ordered by timestamp.

        Args:
            run_id: Analysis run identifier.

        Returns:
            List of log documents sorted ascending by timestamp.
        """
        query = (
            "SELECT * FROM c WHERE c.run_id = @run_id ORDER BY c.timestamp ASC"
        )
        params = [{"name": "@run_id", "value": run_id}]

        items = list(
            self._container.query_items(
                query=query,
                parameters=params,
                partition_key=run_id,
            )
        )
        logger.debug(
            "CosmosDecisionLogClient: fetched %d entries for run=%s", len(items), run_id
        )
        return items

    def get_agent_errors(self, run_id: str) -> list[dict]:
        """
        Fetch only error entries for a run — useful for debugging failed pipelines.
        """
        query = (
            "SELECT * FROM c WHERE c.run_id = @run_id AND c.status = 'error' "
            "ORDER BY c.timestamp ASC"
        )
        params = [{"name": "@run_id", "value": run_id}]

        return list(
            self._container.query_items(
                query=query,
                parameters=params,
                partition_key=run_id,
            )
        )


# Module-level singleton
cosmos_decision_log = CosmosDecisionLogClient()