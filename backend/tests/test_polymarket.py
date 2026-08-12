import pytest

from backend.markets.polymarket import fetch_event, fetch_market, market_to_decimal_odds


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeSession:
    """Maps a URL substring to a canned payload; records every call made."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        for key, payload in self.responses.items():
            if key in url:
                return FakeResponse(payload)
        return FakeResponse([])


SAMPLE_MARKET = {
    "id": "558936",
    "slug": "will-france-win-the-2026-fifa-world-cup-924",
    "question": "Will France win the 2026 FIFA World Cup?",
    "outcomes": '["Yes", "No"]',
    "outcomePrices": '["0.17", "0.83"]',
}


def test_fetch_market_by_slug_passes_slug_param():
    session = FakeSession({"/markets": [SAMPLE_MARKET]})
    market = fetch_market(slug="will-france-win-the-2026-fifa-world-cup-924", session=session)
    assert market["id"] == "558936"
    assert session.calls[0][1] == {"slug": "will-france-win-the-2026-fifa-world-cup-924"}


def test_fetch_market_by_id_passes_id_param():
    session = FakeSession({"/markets": [SAMPLE_MARKET]})
    market = fetch_market(market_id="558936", session=session)
    assert market is not None
    assert session.calls[0][1] == {"id": "558936"}


def test_fetch_market_requires_slug_or_id():
    with pytest.raises(ValueError):
        fetch_market()


def test_fetch_market_returns_none_when_empty():
    session = FakeSession({"/markets": []})
    assert fetch_market(slug="nonexistent", session=session) is None


def test_fetch_event_by_slug_returns_first_result():
    session = FakeSession({"/events": [{"id": "30615", "slug": "world-cup-winner", "markets": [SAMPLE_MARKET]}]})
    event = fetch_event(slug="world-cup-winner", session=session)
    assert event["id"] == "30615"
    assert len(event["markets"]) == 1


def test_fetch_event_by_id_hits_direct_path():
    session = FakeSession({"/events/30615": {"id": "30615", "slug": "world-cup-winner"}})
    event = fetch_event(event_id="30615", session=session)
    assert event["id"] == "30615"
    assert session.calls[0][0].endswith("/events/30615")


def test_market_to_decimal_odds_parses_json_string_fields():
    # Gamma API returns outcomes/outcomePrices as JSON-encoded strings, not lists.
    odds = market_to_decimal_odds(SAMPLE_MARKET)
    assert abs(odds["Yes"] - 1 / 0.17) < 1e-6
    assert abs(odds["No"] - 1 / 0.83) < 1e-6


def test_market_to_decimal_odds_accepts_already_parsed_lists():
    market = {"outcomes": ["Yes", "No"], "outcomePrices": [0.5, 0.5]}
    odds = market_to_decimal_odds(market)
    assert odds == {"Yes": 2.0, "No": 2.0}


def test_market_to_decimal_odds_skips_zero_priced_outcomes():
    market = {"outcomes": ["Yes", "No"], "outcomePrices": ["0", "1"]}
    odds = market_to_decimal_odds(market)
    assert "Yes" not in odds
    assert odds["No"] == 1.0


def test_market_to_decimal_odds_returns_empty_for_market_with_no_prices_yet():
    # Some low-liquidity markets report outcomePrices as None.
    market = {"outcomes": '["Yes", "No"]', "outcomePrices": None}
    assert market_to_decimal_odds(market) == {}
