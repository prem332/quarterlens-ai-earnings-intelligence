"""
Shared test setup, applied before any test module is collected/imported.

Every azure_clients/*.py module builds its client as a module-level
singleton (`kv = KeyVaultClient()`, `openai_client = OpenAIClient()`, etc.),
constructed the moment the module is first imported — including
transitively: importing agents/numeric_validation_agent.py pulls in
azure_clients/openai_client.py, which pulls in key_vault_client.py.
ci.yml's lint-and-test job never runs azure/login and sets no
AZURE_KEY_VAULT_URL, so KeyVaultClient falls back to plain environment
variables (its own documented fallback) — which don't exist either unless
something sets them. Without the dummy values below, importing any
agent/tool module in CI raises ValueError at collection time, before a
single test body runs.

Real SDK client constructors used here (AzureOpenAI, SearchClient, pyodbc's
connection wrapper) don't make network calls at construction — dummy
values are enough to satisfy them. The one exception is
CosmosDecisionLogClient.__init__ (azure_clients/cosmos_client.py), which
calls create_database_if_not_exists()/create_container_if_not_exists() for
real at construction time — pulled in transitively by agents/supervisor.py
-> graph/build_graph.py. Patching azure.cosmos.CosmosClient before that
module is ever imported avoids a real (and failing) network call against a
fake endpoint during test collection.
"""
import os
from unittest.mock import MagicMock

_DUMMY_SECRETS = {
    "AZURE_OPENAI_ENDPOINT": "https://fake.openai.azure.com/",
    "AZURE_OPENAI_KEY": "fake-openai-key",
    "AZURE_OPENAI_DEPLOYMENT_NAME": "fake-deployment",
    "AZURE_OPENAI_DEPLOYMENT_NAME_STANDARD": "fake-deployment-standard",
    "AZURE_SEARCH_ENDPOINT": "https://fake.search.windows.net/",
    "AZURE_SEARCH_ADMIN_KEY": "fake-search-key",
    "AZURE_SQL_SERVER": "fake.database.windows.net",
    "AZURE_SQL_DATABASE": "fake-db",
    "AZURE_SQL_USERNAME": "fake-user",
    "AZURE_SQL_PASSWORD": "fake-password",
    "AZURE_COSMOS_URI": "https://fake.documents.azure.com:443/",
    "AZURE_COSMOS_KEY": "ZmFrZS1jb3Ntb3Mta2V5",
}
for _k, _v in _DUMMY_SECRETS.items():
    os.environ.setdefault(_k, _v)

import azure.cosmos as _azure_cosmos  # noqa: E402

_azure_cosmos.CosmosClient = MagicMock()
