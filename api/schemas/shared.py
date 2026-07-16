from enum import Enum
from typing import Literal


class Company(str, Enum):
    AAPL = "AAPL"
    MSFT = "MSFT"
    NVDA = "NVDA"
    GOOGL = "GOOGL"
    META = "META"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# e.g. "Q1_2025", "Q2_2024"
Quarter = str

AgentName = Literal[
    "retrieval",
    "comparison",
    "sentiment",
    "numeric_validation",
    "report",
]