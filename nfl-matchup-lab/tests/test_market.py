import numpy as np
import pytest

from nfl_lab.features import market as m

METHODS = ["multiplicative", "additive", "power", "shin"]


@pytest.mark.parametrize("method", METHODS)
def test_even_market_devigs_to_half(method):
    q = [m.american_to_prob(-110)] * 2
    assert m.devig(q, method) == pytest.approx([0.5, 0.5], abs=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_devig_sums_to_one_and_keeps_order(method):
    for a, b in [(-175, 145), (-300, 240), (-120, 100), (-1000, 650)]:
        p = m.devig([m.american_to_prob(a), m.american_to_prob(b)], method)
        assert p.sum() == pytest.approx(1.0, abs=1e-12)
        assert p[0] > p[1] > 0


def test_bills_rams_moneyline_examples():
    q = [m.american_to_prob(-175), m.american_to_prob(145)]
    assert m.devig(q, "multiplicative") == pytest.approx([0.6092, 0.3908], abs=5e-5)
    assert m.devig(q, "power") == pytest.approx([0.6166, 0.3834], abs=5e-5)


def test_power_and_shin_shift_more_vig_onto_longshot_than_multiplicative():
    q = [m.american_to_prob(-1000), m.american_to_prob(650)]
    mult, power, shin = (m.devig(q, k)[1] for k in ("multiplicative", "power", "shin"))
    assert power < mult and shin < mult


def test_three_way_market():
    q = np.array([0.5, 0.35, 0.25])
    for k in METHODS:
        assert m.devig(q, k).sum() == pytest.approx(1.0)


def test_american_round_trip():
    for o in (-250, -110, 100, 145, 600):
        back = m.prob_to_american(m.american_to_prob(o))
        assert back == pytest.approx(o) or (abs(o) == 100 and abs(back) == pytest.approx(100))


@pytest.mark.parametrize("cell,expect,kind", [
    ("-110", 110 / 210, "american"), ("+145", 100 / 245, "american"), ("145", 100 / 245, "american"),
    ("52c", 0.52, "exchange"), ("52¢", 0.52, "exchange"), ("0.52", 0.52, "prob"), ("52%", 0.52, "prob"),
    ("", None, "none"), (None, None, "none"), (float("nan"), None, "none"),
])
def test_parse_price(cell, expect, kind):
    p, k = m.parse_price(cell)
    assert k == kind
    assert (p is None and expect is None) or p == pytest.approx(expect)


def test_parse_price_rejects_garbage():
    with pytest.raises(ValueError):
        m.parse_price("abc")
    with pytest.raises(ValueError):
        m.parse_price("55")      # not odds, not a probability


def test_fair_two_way_cases():
    f, note = m.fair_two_way(m.american_to_prob(-115), m.american_to_prob(-105))
    assert note == "two-way" and 0.5 < f < 0.53
    f, note = m.fair_two_way(0.54, None, one_sided_overround=0.045)
    assert note == "one-sided" and f == pytest.approx(0.54 / 1.045)
    f, note = m.fair_two_way(0.52, 0.47)                  # exchange prices under 100%: no vig removed
    assert note == "two-way (no vig)" and f == pytest.approx(0.52 / 0.99)
    assert m.fair_two_way(None, None) == (None, "no price")


def test_min_edge_is_smallest_across_methods():
    ps, po = m.american_to_prob(-140), m.american_to_prob(115)
    e = m.min_edge(0.60, ps, po)
    edges = [0.60 - m.devig([ps, po], k)[0] for k in ("multiplicative", "power", "shin")]
    assert e == pytest.approx(min(edges))
    assert m.min_edge(0.6, None, None) is None
