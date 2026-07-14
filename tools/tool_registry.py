"""
OpenAI function-calling schemas for all QuarterLens tools.
Each schema follows the JSON Schema subset supported by Azure OpenAI function calling.

Usage in agent nodes:
    from tools.tool_registry import TOOLS, dispatch_tool

    # Pass TOOLS to the chat completion call:
    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    # Execute the tool the model selected:
    result = dispatch_tool(tool_name, tool_args_dict)
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Hybrid search (vector + BM25 keyword) over SEC filings and earnings call transcripts "
                "in the quarterlens-filings Azure AI Search index. Use this to retrieve relevant passages "
                "for a given query, optionally filtered by document type, company, or fiscal quarter."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query describing the information needed.",
                    },
                    "doc_type": {
                        "type": "string",
                        "enum": ["10-Q", "10-K", "transcript"],
                        "description": "Filter results to a specific document type. Omit to search all types.",
                    },
                    "company": {
                        "type": "string",
                        "description": "Ticker symbol to filter results (e.g. 'AAPL', 'MSFT'). Omit to search all companies.",
                    },
                    "quarter": {
                        "type": "string",
                        "description": "Fiscal label to filter results (e.g. 'Q2_FY2025'). Omit to search all quarters.",
                    },
                    "top": {
                        "type": "integer",
                        "description": "Number of results to return. Default is 5.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_prior_quarter",
            "description": (
                "Retrieve relevant chunks from a prior quarter for comparison. "
                "Resolves the target quarter by stepping back `quarters_back` periods from "
                "`current_quarter`, then searches the index for matching passages. "
                "Use this in the Comparison Agent to detect language shifts across quarters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {
                        "type": "string",
                        "description": "Ticker symbol (e.g. 'AAPL').",
                    },
                    "current_quarter": {
                        "type": "string",
                        "description": "The quarter being analyzed, in fiscal label format (e.g. 'Q2_FY2025').",
                    },
                    "quarters_back": {
                        "type": "integer",
                        "description": "How many quarters to step back. 1 = immediate prior quarter, 4 = same quarter prior year.",
                        "minimum": 1,
                    },
                    "query": {
                        "type": "string",
                        "description": "The passage or topic to search for in the prior quarter.",
                    },
                    "doc_type": {
                        "type": "string",
                        "enum": ["10-Q", "10-K", "transcript"],
                        "description": "Optionally restrict to a specific document type.",
                    },
                    "top": {
                        "type": "integer",
                        "description": "Number of results to return. Default is 5.",
                        "default": 5,
                    },
                },
                "required": ["company", "current_quarter", "quarters_back", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_finbert",
            "description": (
                "Run FinBERT financial sentiment analysis over transcript text. "
                "Returns positive/negative/neutral scores at both aggregate and per-window level. "
                "Use this in the Sentiment Agent — do NOT ask the LLM to evaluate sentiment directly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The transcript passage or full transcript body to analyze.",
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "Max tokens per analysis window (≤ 512). Default is 400.",
                        "default": 400,
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_metric",
            "description": (
                "Deterministically verify a numeric claim against filed financial facts in Azure SQL. "
                "Fetches typed us-gaap values from `financial_facts` and computes the specified formula. "
                "Never uses LLM arithmetic — use this for all numeric verification in the Numeric Validation Agent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "formula": {
                        "type": "string",
                        "enum": ["yoy_growth", "qoq_growth", "gross_margin", "operating_margin", "net_margin", "ratio"],
                        "description": (
                            "Metric to compute. "
                            "yoy_growth/qoq_growth: percentage change vs prior period (requires prior_fiscal_label). "
                            "gross_margin: (revenue - COGS) / revenue × 100. "
                            "operating_margin: operating income / revenue × 100. "
                            "net_margin: net income / revenue × 100. "
                            "ratio: concept / denominator_concept (requires denominator_concept)."
                        ),
                    },
                    "company": {
                        "type": "string",
                        "description": "Ticker symbol (e.g. 'AAPL').",
                    },
                    "fiscal_label": {
                        "type": "string",
                        "description": "Primary period to compute the metric for (e.g. 'Q2_FY2025').",
                    },
                    "concept": {
                        "type": "string",
                        "description": (
                            "Primary us-gaap concept name from financial_facts "
                            "(e.g. 'Revenues', 'NetIncomeLoss'). "
                            "For margin formulas this is the revenue concept."
                        ),
                    },
                    "prior_fiscal_label": {
                        "type": "string",
                        "description": "Prior period label (e.g. 'Q2_FY2024'). Required for yoy_growth and qoq_growth.",
                    },
                    "denominator_concept": {
                        "type": "string",
                        "description": "Denominator concept name. Required for 'ratio' formula.",
                    },
                    "cogs_concept": {
                        "type": "string",
                        "description": "COGS concept override for gross_margin. Default: 'CostOfRevenue'.",
                    },
                    "operating_income_concept": {
                        "type": "string",
                        "description": "Operating income concept override. Default: 'OperatingIncomeLoss'.",
                    },
                    "net_income_concept": {
                        "type": "string",
                        "description": "Net income concept override. Default: 'NetIncomeLoss'.",
                    },
                },
                "required": ["formula", "company", "fiscal_label", "concept"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Dispatch table — maps function name → callable
# Imported lazily to avoid circular imports at module load time
# ---------------------------------------------------------------------------

def dispatch_tool(name: str, arguments: dict[str, Any]) -> dict:
    """
    Execute a tool by name with the arguments dict the model returned.

    Args:
        name:      Function name as returned in the model's tool_call.
        arguments: Parsed dict of arguments (json.loads the model's arguments string first).

    Returns:
        Tool result dict, ready to be sent back as a tool message.

    Raises:
        ValueError if the tool name is not registered.
    """
    # Import here to keep module-level load fast and avoid circular deps
    from tools.search_documents import search_documents
    from tools.fetch_prior_quarter import fetch_prior_quarter
    from tools.run_finbert import run_finbert
    from tools.calculate_metric import calculate_metric

    _registry = {
        "search_documents": search_documents,
        "fetch_prior_quarter": fetch_prior_quarter,
        "run_finbert": run_finbert,
        "calculate_metric": calculate_metric,
    }

    if name not in _registry:
        raise ValueError(f"Unknown tool '{name}'. Registered tools: {list(_registry.keys())}")

    return _registry[name](**arguments)


def dispatch_tool_from_json(name: str, arguments_json: str) -> dict:
    """
    Convenience wrapper: parses the JSON string the model returns before dispatching.
    """
    arguments = json.loads(arguments_json)
    return dispatch_tool(name, arguments)