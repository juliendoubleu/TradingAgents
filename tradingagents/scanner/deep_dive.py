"""Hand the top candidates to the existing LLM agent graph, so its verdict stands beside the scanner's.

The scan itself needs no model. This step is optional: for each given signal it runs
``TradingAgentsGraph.propagate`` and keeps the 5-tier rating. A ticker that fails is
recorded with its error and does not stop the others.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field

from tradingagents.agents.rating import run_rating
from tradingagents.dataflows.symbols import crypto_base
from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.scanner.signals import Signal

logger = logging.getLogger(__name__)

STOCK_ANALYSTS = ("market", "social", "news", "fundamentals")
CRYPTO_ANALYSTS = ("market", "social", "news")      # no fundamentals for a coin


@dataclass
class DeepDiveResult:
    ratings: dict[str, str] = field(default_factory=dict)    # ticker -> Buy/Overweight/Hold/...
    errors: dict[str, str] = field(default_factory=dict)     # ticker -> why it failed


def check_llm_credentials(config: dict) -> None:
    """Raise ``RuntimeError`` naming the missing key before any graph is built.

    Checks the default provider and each tier's own provider. A provider that needs no
    key (a local runtime, Bedrock's credential chain) passes.
    """
    providers = {config.get("llm_provider"), config.get("deep_think_provider"),
                 config.get("quick_think_provider")} - {None, ""}
    for provider in sorted(providers):
        env = get_api_key_env(str(provider))
        if env and not os.environ.get(env, "").strip():
            raise RuntimeError(
                f"--deep needs an LLM and {env} is not set for provider '{provider}'. "
                f"Set it (for example in .env), or choose another provider with "
                f"TRADINGAGENTS_LLM_PROVIDER."
            )


def deep_dive(
    signals: list[Signal],
    as_of_date: str,
    config: dict,
    graph_factory: Callable[[tuple[str, ...], dict], object] | None = None,
) -> DeepDiveResult:
    """Run the agent graph on each signal's ticker and collect the rating it ends on.

    ``graph_factory(analysts, config)`` builds a graph; the default builds a
    ``TradingAgentsGraph``. It is injectable so tests need no model.
    """
    if graph_factory is None:
        def graph_factory(analysts, cfg):
            from tradingagents.graph.trading_graph import TradingAgentsGraph
            return TradingAgentsGraph(list(analysts), config=cfg)

    result = DeepDiveResult()
    graphs: dict[bool, object] = {}
    for sig in signals:
        crypto = crypto_base(sig.symbol) is not None
        try:
            if crypto not in graphs:
                graphs[crypto] = graph_factory(
                    CRYPTO_ANALYSTS if crypto else STOCK_ANALYSTS, config)
            final_state, _signal = graphs[crypto].propagate(
                sig.symbol, as_of_date, "crypto" if crypto else "stock")
            result.ratings[sig.symbol] = run_rating(final_state)
        except Exception as exc:  # one failing ticker must not end the rest
            logger.warning("Deep dive for %s failed: %s", sig.symbol, exc)
            result.errors[sig.symbol] = f"{type(exc).__name__}: {exc}"
    return result
