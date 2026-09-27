"""Latency-arbitrage research bot for Polymarket crypto "Up or Down" markets.

Layers (each one only depends on the ones above it):

    config / clock          settings with hard bounds, wall vs replay clock
    model                   P(fair) digital-option pricing, realized vol, basis
    data                    WebSocket feeds, parsers, order books, Gamma discovery,
                            reference prices, tick recorder / replay
    signal                  P(fair) vs P(market) after fees -> signal records
    reporting               CSV sinks, calibration / shadow-signal analysis

Execution and risk management are intentionally NOT part of this phase.
"""

__version__ = "0.1.0"
