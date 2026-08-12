import json

import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"


def fetch_market(slug=None, market_id=None, session=None):
    """
    Fetches a single market from Polymarket's public Gamma API (read-only,
    no auth required). Pass slug OR market_id. `session` accepts anything
    with a requests-compatible .get() -- defaults to the requests module
    itself; tests inject a fake to avoid depending on network access.
    """
    if not slug and not market_id:
        raise ValueError("must provide slug or market_id")

    http = session or requests
    params = {"slug": slug} if slug else {"id": market_id}
    resp = http.get(f"{GAMMA_BASE}/markets", params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data[0] if data else None


def fetch_event(event_id=None, slug=None, session=None):
    """
    Fetches an event (a group of related markets, e.g. all 2026 World Cup
    "will X win it all" markets under one "World Cup Winner" event) by id
    or slug. Returns the event dict, including its child `markets` list.
    """
    if not event_id and not slug:
        raise ValueError("must provide event_id or slug")

    http = session or requests
    if event_id:
        resp = http.get(f"{GAMMA_BASE}/events/{event_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()

    resp = http.get(f"{GAMMA_BASE}/events", params={"slug": slug}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data[0] if data else None


def market_to_decimal_odds(market):
    """
    Converts a Polymarket market's outcome prices (each already a
    probability in [0, 1]) into decimal odds (1/price) -- the same
    representation backend.markets.devig operates on. Returns
    {outcome_label: decimal_odds}, skipping any zero-priced (impossible)
    outcome to avoid a division by zero. Returns {} for a market with no
    prices yet (some low-liquidity markets report outcomePrices as None).
    """
    outcomes = market.get("outcomes")
    prices = market.get("outcomePrices")
    if not outcomes or not prices:
        return {}
    if isinstance(outcomes, str):
        outcomes = json.loads(outcomes)
    if isinstance(prices, str):
        prices = json.loads(prices)

    decimal_odds = {}
    for outcome, price in zip(outcomes, prices):
        price = float(price)
        if price > 0:
            decimal_odds[outcome] = 1.0 / price
    return decimal_odds
